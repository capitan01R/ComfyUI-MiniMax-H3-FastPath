from .middle_cache import MiniMaxH3EulerMiddleCache


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3EulerMiddleCache": MiniMaxH3EulerMiddleCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3EulerMiddleCache": "MiniMax H3 FastPath Euler Middle Cache",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
