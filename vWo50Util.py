from typing import Any, Mapping
import torch
import math


"""A special class that is always equal in not equal comparisons."""
class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False


anyType = AnyType("*")


class GetType:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"any": (anyType, {})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "获取类型信息"

    def process(self, any: Any) -> tuple[str]:
        return (str(type(any)),)


class GetElementType:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"container": (anyType, {})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "获取下标为0的元素的类型信息"

    def process(self, container: Any) -> tuple[str]:
        return (str(type(container.get(0, None))),)


class GetElementAtIndex:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"container": (anyType, {}), "index": ("INT", {"min": 0})}}

    RETURN_TYPES = (AnyType("elememt"),)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "根据下标获取容器内的元素"

    def process(self, container: Any, index: int) -> tuple[Any]:
        return (container[index],)

    @staticmethod
    def get_model_names():
        return ["model1", "model2", "model3"]


class GetMaskForIPAdapterImage:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"image": ("IMAGE", {})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "使用IPAdapter时，图片都规定是正方形（一般1024x1024）的，当我们的图片是长方形时，可以以scale to " \
                  "fit的方式调整图片大小然后叠加在一个1024x1024的空白图片上，然后把实际图片内容作为蒙版传递给IPAdapter。这个节点就是用于获取这个蒙版"

    def process(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("输入必须是 torch.Tensor 类型")

        if image.ndimension() == 4:
            # 形状为 (B, H, W, C)
            batch, height, width, channels = image.shape
        elif image.ndimension() == 3:
            # 形状为 (C, H, W)
            channels, height, width = image.shape
        elif image.ndimension() == 2:
            # 形状为 (H, W)
            height, width = image.shape
        else:
            raise ValueError("image.ndimension错误")

        x, y, mask_width, mask_height = 0, 0, 0, 0
        if width == height:
            x, y = 0, 0
            mask_width, mask_height = 1024, 1024
        elif width > height:
            x = 0
            mask_width = 1024
            mask_height = int(1024.0 / width * height)
            y = math.ceil((1024 - mask_height) / 2.0)
        else:
            y = 0
            mask_height = 1024
            mask_width = int(1024.0 / height * width)
            x = math.ceil((1024 - mask_width) / 2.0)

        # 创建一个全零的张量，形状为 (1, 1024, 1024)
        mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        # 设置指定区域为 1
        mask[0, y:y+mask_height, x:x+mask_width] = 1.0

        return (mask,)


class GetInfoOfTorchTensor:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"tensor_obj": (anyType, {})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "获取torch.Tensor对象的信息"

    def process(self, tensor_obj: Any) -> tuple[str]:
        return (f"dimension: {str(tensor_obj.ndimension())}, shape: {str(tensor_obj.shape)}, dtype: {str(tensor_obj.dtype)}, device: {str(tensor_obj.device)}",)


class ExecutePythonCode:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {},
                "optional": {
                        "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                        "argument1": (anyType,),
                        "argument2": (anyType,)
                    }
                }

    RETURN_TYPES = (AnyType("result"),)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    OUTPUT_NODE = True
    DESCRIPTION = "动态执行python代码"

    def process(self, code: str, argument1: Any = None, argument2: Any = None) -> tuple[Any]:
        result = ""
        local_vars = {"argument1": argument1, "argument2": argument2}
        exec(code, {}, local_vars)
        result = local_vars.get("result", None)
        return (result,)


class ExecutePythonCodeReturnInt:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {},
                "optional": {
                        "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                        "argument1": (anyType,),
                        "argument2": (anyType,)
                    }
                }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    OUTPUT_NODE = True
    DESCRIPTION = "动态执行python代码"

    def process(self, code: str, argument1: Any = None, argument2: Any = None) -> tuple[Any]:
        result = ""
        local_vars = {"argument1": argument1, "argument2": argument2}
        exec(code, {}, local_vars)
        result = local_vars.get("result", None)
        return (int(result),)


NODE_CLASS_MAPPINGS = {
    "VWo50 Get Type": GetType,
    "VWo50 Get Element Type": GetElementType,
    "VWo50 Get Element At Index": GetElementAtIndex,
    "VWo50 Get Mask For IPAdapter Image": GetMaskForIPAdapterImage,
    "VWo50 Get Info of torch Tensor": GetInfoOfTorchTensor,
    "VWo50 Execute Python Code": ExecutePythonCode,
    "VWo50 Execute Python Code Return Int": ExecutePythonCodeReturnInt,
}
