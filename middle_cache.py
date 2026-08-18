from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import torch

import comfy.patcher_extension

from .common import (
    CACHE_ATTACHMENT_KEY,
    CACHE_BINDING_KEY,
    CACHE_CALL_KEY,
    CACHE_WRAPPER_KEY,
    LORA_SIGNATURE_KEY,
    normalized_items,
)


LOG = logging.getLogger(__name__)
RUN_ID_KEY = "minimax_h3_fastpath_cache_run_id"
STEP_ID_KEY = "minimax_h3_fastpath_cache_step_id"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    first_middle_block: int
    last_middle_block: int
    reuse_threshold: float
    max_consecutive_reuses: int
    cache_device: str
    require_fastpath_schedule: bool
    strict_branch_identity: bool
    suppress_candidate_prefetch: bool
    verbose: bool

    @property
    def middle_block_count(self) -> int:
        return self.last_middle_block - self.first_middle_block + 1


@dataclass(slots=True)
class BranchState:
    topology: tuple[Any, ...] | None = None
    lora_signature: tuple[Any, ...] | None = None
    residual: torch.Tensor | None = None
    prefix_probe: torch.Tensor | None = None
    consecutive_reuses: int = 0
    last_step_id: int = -1

    def reset(self, topology=None, lora_signature=None) -> None:
        self.topology = topology
        self.lora_signature = lora_signature
        self.residual = None
        self.prefix_probe = None
        self.consecutive_reuses = 0
        self.last_step_id = -1


@dataclass(slots=True)
class CacheCall:
    runtime: "MiddleCacheRuntime"
    run_id: int
    step_id: int
    branch_key: tuple[Any, ...] | None
    lora_signature: tuple[Any, ...] | None
    eligible: bool
    reason: str
    prefetch_suppressed: bool = False
    middle_started: bool = False
    reuse: bool = False
    completed: bool = False
    state: BranchState | None = None
    start_hidden: torch.Tensor | None = None
    current_probe: torch.Tensor | None = None
    feature_delta: float | None = None


@dataclass(slots=True)
class CacheBinding:
    runtime: "MiddleCacheRuntime"


class MiddleCacheRuntime:
    """Run-local residual cache for a contiguous range of H3 transformer blocks."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.lock = threading.RLock()
        self.run_serial = 0
        self.active_run_id: int | None = None
        self.active_step_id: int | None = None
        self.supported_sampler = False
        self.branch_states: dict[tuple[Any, ...], BranchState] = {}
        self.step_occurrences: dict[tuple[Any, ...], int] = {}
        self.disabled_reason: str | None = None
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict[str, int]:
        return {
            "model_calls": 0,
            "cached_full_calls": 0,
            "reused_calls": 0,
            "skipped_blocks": 0,
            "strength_resets": 0,
            "gate_rejections": 0,
            "identity_fallbacks": 0,
            "prefetch_suppressed_calls": 0,
        }

    def start_run(self, sampler: Any) -> int:
        with self.lock:
            self.clear()
            self.run_serial += 1
            self.active_run_id = self.run_serial
            self.supported_sampler = sampler_name(sampler) == "sample_euler"
            return self.run_serial

    def clear(self) -> None:
        self.branch_states.clear()
        self.step_occurrences.clear()
        self.active_step_id = None
        self.disabled_reason = None
        self.stats = self._new_stats()

    def end_run(self, run_id: int) -> dict[str, int]:
        with self.lock:
            snapshot = dict(self.stats)
            if self.active_run_id == run_id:
                self.clear()
                self.active_run_id = None
                self.supported_sampler = False
            return snapshot

    def begin_step(self, timestep: Any) -> int:
        del timestep
        with self.lock:
            if self.active_run_id is None:
                return -1
            self.active_step_id = 0 if self.active_step_id is None else self.active_step_id + 1
            self.step_occurrences.clear()
            return self.active_step_id

    def new_model_call(self, options: dict[str, Any]) -> CacheCall:
        with self.lock:
            self.stats["model_calls"] += 1
            run_id = int(options.get(RUN_ID_KEY, -1))
            step_id = int(options.get(STEP_ID_KEY, -1))

            cond_labels = normalized_items(options.get("cond_or_uncond"))
            uuids = normalized_items(options.get("uuids"))
            base_branch: tuple[Any, ...] | None
            if self.config.strict_branch_identity and not cond_labels:
                base_branch = None
            else:
                base_branch = (cond_labels or ("single",), uuids)

            if base_branch is None:
                branch_key = None
                self.stats["identity_fallbacks"] += 1
            else:
                occurrence = self.step_occurrences.get(base_branch, 0)
                self.step_occurrences[base_branch] = occurrence + 1
                branch_key = (*base_branch, occurrence)

            raw_signature = options.get(LORA_SIGNATURE_KEY)
            lora_signature = tuple(raw_signature) if raw_signature is not None else None

            eligible = True
            reason = "eligible"
            if self.active_run_id != run_id or self.active_step_id != step_id:
                eligible, reason = False, "outside active Euler step"
            elif not self.supported_sampler:
                eligible, reason = False, "sampler is not stock Euler"
            elif self.disabled_reason is not None:
                eligible, reason = False, self.disabled_reason
            elif options.get("multigpu_thread_device") is not None:
                eligible, reason = False, "multi-GPU calls use the exact path"
            elif branch_key is None:
                eligible, reason = False, "CFG branch identity unavailable"
            elif self.config.require_fastpath_schedule and lora_signature is None:
                eligible, reason = False, "FastPath scheduled-LoRA signature unavailable"
            elif lora_signature is None:
                lora_signature = (("static-model", 1.0),)

            prefetch_suppressed = False
            if eligible and self.config.suppress_candidate_prefetch:
                state = self.branch_states.get(branch_key)
                prefetch_suppressed = bool(
                    state is not None
                    and state.residual is not None
                    and state.lora_signature == lora_signature
                    and state.consecutive_reuses < self.config.max_consecutive_reuses
                    and state.last_step_id != step_id
                )
                if prefetch_suppressed:
                    self.stats["prefetch_suppressed_calls"] += 1

            return CacheCall(
                runtime=self,
                run_id=run_id,
                step_id=step_id,
                branch_key=branch_key,
                lora_signature=lora_signature,
                eligible=eligible,
                reason=reason,
                prefetch_suppressed=prefetch_suppressed,
            )

    @staticmethod
    def _topology(hidden: torch.Tensor, args: dict[str, Any]) -> tuple[Any, ...]:
        segments = tuple(tuple(int(value) for value in segment) for segment in args["mod_segments"])
        return (
            tuple(hidden.shape),
            str(hidden.dtype),
            str(hidden.device),
            segments,
            tuple(args["t_emb"].shape),
            tuple(args["rope_freqs"].shape),
        )

    @staticmethod
    def _probe(hidden: torch.Tensor, max_rows: int = 96, max_columns: int = 128) -> torch.Tensor:
        flat = hidden.detach().reshape(-1, hidden.shape[-1])
        row_stride = max(1, math.ceil(flat.shape[0] / max_rows))
        col_stride = max(1, math.ceil(flat.shape[1] / max_columns))
        return flat[::row_stride, ::col_stride][:max_rows, :max_columns].float().clone()

    @staticmethod
    def _probe_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
        denominator = previous.abs().mean().clamp_min_(1.0e-6)
        return float(((current - previous).abs().mean() / denominator).item())

    def begin_middle(self, call: CacheCall, hidden: torch.Tensor, args: dict[str, Any]) -> None:
        call.middle_started = True
        if not call.eligible:
            return

        topology = self._topology(hidden, args)
        probe = self._probe(hidden)
        call.current_probe = probe

        with self.lock:
            state = self.branch_states.setdefault(call.branch_key, BranchState())
            call.state = state

            if state.topology != topology:
                state.reset(topology=topology, lora_signature=call.lora_signature)
                call.reason = "topology changed or cache empty"
            elif state.lora_signature != call.lora_signature:
                state.reset(topology=topology, lora_signature=call.lora_signature)
                self.stats["strength_resets"] += 1
                call.reason = "LoRA strength signature changed"

            can_reuse = (
                state.residual is not None
                and state.prefix_probe is not None
                and state.consecutive_reuses < self.config.max_consecutive_reuses
                and state.last_step_id != call.step_id
            )
            if can_reuse:
                delta = self._probe_delta(probe, state.prefix_probe)
                call.feature_delta = delta
                if delta <= self.config.reuse_threshold:
                    call.reuse = True
                    state.consecutive_reuses += 1
                    state.last_step_id = call.step_id
                    self.stats["reused_calls"] += 1
                    self.stats["skipped_blocks"] += self.config.middle_block_count
                    call.reason = "middle residual reused"
                    return
                self.stats["gate_rejections"] += 1
                call.reason = f"prefix feature delta {delta:.5f} exceeded gate"
            elif state.residual is not None and state.consecutive_reuses >= self.config.max_consecutive_reuses:
                call.reason = "maximum consecutive reuse reached"

        try:
            call.start_hidden = hidden.detach().clone()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            with self.lock:
                self.disabled_reason = "cache allocation ran out of memory"
                self.branch_states.clear()
            call.eligible = False
            call.reason = self.disabled_reason
            LOG.warning("MiniMax H3 FastPath cache disabled for this run: %s", error)

    def apply_reuse(self, call: CacheCall, hidden: torch.Tensor) -> torch.Tensor:
        state = call.state
        if state is None or state.residual is None:
            raise RuntimeError("FastPath cache selected reuse without a residual")
        residual = state.residual
        if residual.device != hidden.device or residual.dtype != hidden.dtype:
            residual = residual.to(device=hidden.device, dtype=hidden.dtype, non_blocking=True)
        hidden.add_(residual)
        return hidden

    def finish_middle(self, call: CacheCall, output: torch.Tensor) -> None:
        if not call.eligible or call.reuse or call.start_hidden is None or call.state is None:
            call.completed = True
            return
        try:
            residual = output.detach().clone()
            residual.sub_(call.start_hidden)
            if self.config.cache_device == "cpu":
                residual = residual.to(device="cpu")
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            with self.lock:
                self.disabled_reason = "cache residual allocation ran out of memory"
                self.branch_states.clear()
            call.completed = True
            LOG.warning("MiniMax H3 FastPath cache disabled for this run: %s", error)
            return

        with self.lock:
            call.state.residual = residual
            call.state.prefix_probe = call.current_probe
            call.state.consecutive_reuses = 0
            call.state.last_step_id = call.step_id
            self.stats["cached_full_calls"] += 1
        call.start_hidden = None
        call.completed = True

    def finish_model_call(self, call: CacheCall) -> None:
        call.start_hidden = None
        call.current_probe = None


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


def _binding_from_model_options(model_options: dict[str, Any] | None) -> CacheBinding | None:
    binding = (model_options or {}).get(CACHE_BINDING_KEY)
    return binding if isinstance(binding, CacheBinding) else None


def outer_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None:
        return executor(*args, **kwargs)

    sampler = args[2]
    runtime = binding.runtime
    run_id = runtime.start_run(sampler)
    if runtime.config.verbose:
        LOG.warning(
            "MiniMax H3 FastPath cache run start: sampler=%s supported=%s middle=%d..%d",
            sampler_name(sampler),
            runtime.supported_sampler,
            runtime.config.first_middle_block,
            runtime.config.last_middle_block,
        )
    try:
        return executor(*args, **kwargs)
    finally:
        snapshot = dict(runtime.stats)
        if runtime.config.verbose or snapshot["reused_calls"]:
            LOG.warning(
                "MiniMax H3 FastPath cache summary: reused_calls=%d cached_full_calls=%d "
                "skipped_blocks=%d gate_rejections=%d strength_resets=%d",
                snapshot["reused_calls"],
                snapshot["cached_full_calls"],
                snapshot["skipped_blocks"],
                snapshot["gate_rejections"],
                snapshot["strength_resets"],
            )
        runtime.end_run(run_id)


def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None or binding.runtime.active_run_id is None:
        return executor(x, timestep, model_options or {}, seed)

    runtime = binding.runtime
    step_id = runtime.begin_step(timestep)
    copied = dict(model_options or {})
    transformer_options = dict(copied.get("transformer_options") or {})
    copied["transformer_options"] = transformer_options
    transformer_options[CACHE_BINDING_KEY] = binding
    transformer_options[RUN_ID_KEY] = runtime.active_run_id
    transformer_options[STEP_ID_KEY] = step_id
    return executor(x, timestep, copied, seed)


def diffusion_model_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = dict(transformer_options or {})
    binding = options.get(CACHE_BINDING_KEY)
    if not isinstance(binding, CacheBinding):
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    call = binding.runtime.new_model_call(options)
    if call.prefetch_suppressed:
        # Dynamic ModelPatcher prefetch otherwise streams every middle block even
        # when its replacement returns without executing the block. Candidate
        # calls turn it off before MiniMax constructs the local prefetch queue.
        options["prefetch_dynamic_vbars"] = False
    options[CACHE_CALL_KEY] = call
    try:
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    finally:
        binding.runtime.finish_model_call(call)


def _execute_previous(previous_patch, args, extra_options):
    if previous_patch is None:
        result = extra_options["original_block"](args)
    else:
        result = previous_patch(args, extra_options)
    if not isinstance(result, dict) or "img" not in result:
        raise TypeError("A MiniMax H3 double_block replacement must return a dict containing 'img'")
    return result


def make_middle_block_patch(block_index: int, config: CacheConfig, previous_patch=None):
    is_first = block_index == config.first_middle_block
    is_last = block_index == config.last_middle_block

    def middle_block_patch(args, extra_options):
        call = (args.get("transformer_options") or {}).get(CACHE_CALL_KEY)
        if not isinstance(call, CacheCall):
            return _execute_previous(previous_patch, args, extra_options)

        hidden = args["img"]
        if is_first:
            call.runtime.begin_middle(call, hidden, args)
            if call.reuse:
                return {"img": call.runtime.apply_reuse(call, hidden)}
        elif call.reuse:
            return {"img": hidden}

        result = _execute_previous(previous_patch, args, extra_options)
        if is_last:
            call.runtime.finish_middle(call, result["img"])
        return result

    return middle_block_patch


def model_clone_callback(source_model: Any, cloned_model: Any) -> None:
    source_binding = _binding_from_model_options(getattr(source_model, "model_options", None))
    if source_binding is None:
        return
    cloned_model.model_options[CACHE_BINDING_KEY] = CacheBinding(
        MiddleCacheRuntime(source_binding.runtime.config)
    )


def _existing_dit_replacements(model) -> dict[Any, Any]:
    return (
        model.model_options.get("transformer_options", {})
        .get("patches_replace", {})
        .get("dit", {})
    )


def _install_runtime(model, runtime: MiddleCacheRuntime) -> None:
    model.model_options[CACHE_BINDING_KEY] = CacheBinding(runtime)
    wrappers = comfy.patcher_extension.WrappersMP
    callbacks = comfy.patcher_extension.CallbacksMP

    if not model.get_wrappers(wrappers.OUTER_SAMPLE, CACHE_WRAPPER_KEY):
        model.add_wrapper_with_key(wrappers.OUTER_SAMPLE, CACHE_WRAPPER_KEY, outer_sample_wrapper)
    if not model.get_wrappers(wrappers.PREDICT_NOISE, CACHE_WRAPPER_KEY):
        model.add_wrapper_with_key(wrappers.PREDICT_NOISE, CACHE_WRAPPER_KEY, predict_noise_wrapper)
    if not model.get_wrappers(wrappers.DIFFUSION_MODEL, CACHE_WRAPPER_KEY):
        model.add_wrapper_with_key(
            wrappers.DIFFUSION_MODEL,
            CACHE_WRAPPER_KEY,
            diffusion_model_wrapper,
        )
    if not model.get_callbacks(callbacks.ON_CLONE, CACHE_WRAPPER_KEY):
        model.add_callback_with_key(
            callbacks.ON_CLONE,
            CACHE_WRAPPER_KEY,
            model_clone_callback,
        )


class MiniMaxH3EulerMiddleCache:
    """Keep H3 prefix/suffix exact and cache only a configurable middle block range."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "prefix_blocks": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 48,
                        "step": 1,
                        "tooltip": "Exact blocks run before the cached middle on every model call.",
                    },
                ),
                "suffix_blocks": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 48,
                        "step": 1,
                        "tooltip": "Exact final blocks run after the cached middle on every model call.",
                    },
                ),
                "reuse_threshold": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.005,
                        "tooltip": "Maximum relative change in a sampled prefix feature before reuse is rejected.",
                    },
                ),
                "max_consecutive_reuses": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "One gives FULL/PART/FULL for three calls at the same LoRA strength.",
                    },
                ),
                "cache_device": (
                    ["gpu", "cpu"],
                    {
                        "tooltip": "GPU is fastest. CPU saves VRAM but transfers one full hidden residual on reuse.",
                    },
                ),
                "require_fastpath_schedule": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Refuse approximation unless the matching FastPath scheduled-LoRA signature is present.",
                    },
                ),
                "strict_branch_identity": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use the exact path if ComfyUI does not provide CFG branch labels.",
                    },
                ),
                "suppress_candidate_prefetch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "On likely reuse calls, stop dynamic INT8 weight prefetch so skipped blocks are not streamed anyway.",
                    },
                ),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/FastPath"
    DESCRIPTION = (
        "Stock-Euler-only, LoRA-strength-aware H3 cache. Runs a prefix and suffix exactly; on accepted "
        "calls it reuses only the residual of the middle blocks. Cache state is isolated per CFG branch."
    )

    def patch(
        self,
        model,
        enabled,
        prefix_blocks,
        suffix_blocks,
        reuse_threshold,
        max_consecutive_reuses,
        cache_device,
        require_fastpath_schedule,
        strict_branch_identity,
        suppress_candidate_prefetch,
        verbose,
    ):
        patched = model.clone()
        if not enabled:
            return (patched,)
        if patched.get_attachment(CACHE_ATTACHMENT_KEY) is not None:
            raise RuntimeError("MiniMax H3 FastPath middle cache is already installed on this MODEL")
        if patched.model_options.get("spectrum_h3_binding") is not None:
            raise RuntimeError(
                "Do not stack Spectrum forecasting with FastPath middle caching. Disable Spectrum and use one approximation method."
            )

        try:
            diffusion_model = patched.get_model_object("diffusion_model")
        except Exception as error:
            raise TypeError("MODEL does not expose a diffusion_model") from error
        if type(diffusion_model).__name__ != "MiniMaxH3Model" or not hasattr(diffusion_model, "blocks"):
            raise TypeError("MiniMax H3 FastPath middle cache only supports the native MiniMaxH3Model")

        block_count = len(diffusion_model.blocks)
        prefix_blocks = int(prefix_blocks)
        suffix_blocks = int(suffix_blocks)
        if prefix_blocks + suffix_blocks >= block_count:
            raise ValueError(
                f"prefix_blocks + suffix_blocks must be below the model's {block_count} blocks"
            )
        first_middle = prefix_blocks
        last_middle = block_count - suffix_blocks - 1

        existing = dict(_existing_dit_replacements(patched))
        if ("block_loop", 0) in existing:
            raise RuntimeError(
                "A whole H3 block-loop replacement is already installed. Remove the older MiniMax H3 cache before FastPath."
            )

        config = CacheConfig(
            first_middle_block=first_middle,
            last_middle_block=last_middle,
            reuse_threshold=float(reuse_threshold),
            max_consecutive_reuses=int(max_consecutive_reuses),
            cache_device=str(cache_device),
            require_fastpath_schedule=bool(require_fastpath_schedule),
            strict_branch_identity=bool(strict_branch_identity),
            suppress_candidate_prefetch=bool(suppress_candidate_prefetch),
            verbose=bool(verbose),
        )
        runtime = MiddleCacheRuntime(config)
        _install_runtime(patched, runtime)

        for block_index in range(first_middle, last_middle + 1):
            previous = existing.get(("double_block", block_index))
            patched.set_model_patch_replace(
                make_middle_block_patch(block_index, config, previous),
                "dit",
                "double_block",
                block_index,
            )

        patched.set_attachments(
            CACHE_ATTACHMENT_KEY,
            {
                "prefix_blocks": prefix_blocks,
                "middle_blocks": config.middle_block_count,
                "suffix_blocks": suffix_blocks,
                "reuse_threshold": config.reuse_threshold,
                "max_consecutive_reuses": config.max_consecutive_reuses,
                "cache_device": config.cache_device,
            },
        )
        LOG.info(
            "MiniMax H3 FastPath middle cache installed: %d exact prefix / %d cached middle / %d exact suffix",
            prefix_blocks,
            config.middle_block_count,
            suffix_blocks,
        )
        return (patched,)


__all__ = [
    "BranchState",
    "CacheBinding",
    "CacheCall",
    "CacheConfig",
    "MiddleCacheRuntime",
    "MiniMaxH3EulerMiddleCache",
    "diffusion_model_wrapper",
    "make_middle_block_patch",
    "model_clone_callback",
    "outer_sample_wrapper",
    "predict_noise_wrapper",
]
