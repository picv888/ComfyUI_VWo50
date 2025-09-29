from .vwo50 import NODE_CLASS_MAPPINGS as util_NCM


NODE_CLASS_MAPPINGS = {
    **util_NCM,
}


def remove_cm_prefix(node_mapping: str) -> str:
    if node_mapping.startswith("VWo50 "):
        return node_mapping[6:]
    return node_mapping


NODE_DISPLAY_NAME_MAPPINGS = {key: remove_cm_prefix(key) for key in NODE_CLASS_MAPPINGS}
