from .vWo50_util import NODE_CLASS_MAPPINGS as util_NCM
from .vWo50_type import NODE_CLASS_MAPPINGS as type_NCM

NODE_CLASS_MAPPINGS = {
    **util_NCM,
    **type_NCM,
}


def remove_prefix(node_mapping: str) -> str:
    if node_mapping.startswith("VWo50 "):
        return node_mapping[6:]
    return node_mapping


NODE_DISPLAY_NAME_MAPPINGS = {key: remove_prefix(key) for key in NODE_CLASS_MAPPINGS}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
