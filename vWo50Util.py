from typing import Any, Mapping
import torch
import math


class AnyType(str):
    """
    一个特殊的字符串类，其主要目的是在“不等于”比较中总是返回 False，在“等于”比较中总是返回 True，即使与任何其他对象进行比较。
    """
    def __eq__(self, __value):
        return True

    def __ne__(self, __value):
        return False


anyType = AnyType("*")


class GetType:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"any_object": (anyType, {})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "获取类型信息"

    def process(self, any_object: Any) -> tuple[str]:
        return (str(type(any_object)),)


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
        mask[0, y:y + mask_height, x:x + mask_width] = 1.0

        return (mask,)


class GetMaskForImageScaleToFitRect:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"image": ("IMAGE", {}), "rect_width": ("INT", {}), "rect_height": ("INT", {})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    DESCRIPTION = "在一个rectWidth x rectHeight的矩形框内，把一个图片以scale to fit的方式调整大小时" \
                  "获取这个图片矩形框的蒙版"

    def process(self, image: torch.Tensor, rect_width: int, rect_height: int) -> tuple[torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("输入必须是 torch.Tensor 类型")

        # 获取图片的宽高
        if image.ndimension() == 4:
            # 形状为 (B, H, W, C)
            batch, image_height, image_width, channels = image.shape
        elif image.ndimension() == 3:
            # 形状为 (C, H, W)
            channels, image_height, image_width = image.shape
        elif image.ndimension() == 2:
            # 形状为 (H, W)
            image_height, image_width = image.shape
        else:
            raise ValueError("image.ndimension错误")

        x, y, mask_width, mask_height = 0, 0, 0, 0
        image_ratio = image_width / image_height
        rect_ratio = rect_width / rect_height
        if image_ratio == rect_ratio:
            x, y = 0, 0
            mask_width, mask_height = rect_width, rect_height
        elif image_ratio > rect_ratio:
            x = 0
            mask_width = rect_width
            mask_height = int(mask_width / image_width * image_height)
            y = math.ceil((rect_height - mask_height) / 2.0)
        else:
            y = 0
            mask_height = rect_height
            mask_width = int(mask_height / image_height * image_width)
            x = math.ceil((rect_width - mask_width) / 2.0)

        # 创建一个全零的张量，形状为 (1, 1024, 1024)
        mask = torch.zeros((1, rect_height, rect_width), dtype=torch.float32)
        # 设置指定区域为 1
        mask[0, y:y + mask_height, x:x + mask_width] = 1.0

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
        return (
        f"dimension: {str(tensor_obj.ndimension())}, shape: {str(tensor_obj.shape)}, dtype: {str(tensor_obj.dtype)}, device: {str(tensor_obj.device)}",)


class ExecutePythonCode:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {},
                "optional": {
                    "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                    "argument1": (anyType,),
                    "argument2": (anyType,),
                    "argument3": (anyType,)
                }
                }

    RETURN_TYPES = (AnyType("result1"), AnyType("result2"))
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    OUTPUT_NODE = True
    DESCRIPTION = "动态执行python代码"

    def process(self, code: str, argument1: Any = None, argument2: Any = None, argument3: Any = None) -> tuple[Any, Any]:
        result1 = ""
        result2 = ""
        local_vars = {"argument1": argument1, "argument2": argument2, "argument3": argument3}
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", None)
        result2 = local_vars.get("result2", None)
        return result1, result2


class ExecutePythonCodeReturnInt:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {},
                "optional": {
                    "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                    "argument1": (anyType,),
                    "argument2": (anyType,),
                    "argument3": (anyType,)
                }
                }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    OUTPUT_NODE = True
    DESCRIPTION = "动态执行python代码"

    def process(self, code: str, argument1: Any = None, argument2: Any = None, argument3: Any = None) -> tuple[int, int]:
        result1 = ""
        result2 = ""
        local_vars = {"argument1": argument1, "argument2": argument2, "argument3": argument3}
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", None)
        result2 = local_vars.get("result2", None)
        return int(result1), int(result2)


class ExecutePythonCodeReturnFloat:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {},
                "optional": {
                    "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                    "argument1": (anyType,),
                    "argument2": (anyType,),
                    "argument3": (anyType,)
                }
                }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Util"
    OUTPUT_NODE = True
    DESCRIPTION = "动态执行python代码"

    def process(self, code: str, argument1: Any = None, argument2: Any = None, argument3: Any = None) -> tuple[float, float]:
        result1 = ""
        result2 = ""
        local_vars = {"argument1": argument1, "argument2": argument2, "argument3": argument3}
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", 'nan')
        result2 = local_vars.get("result2", 'nan')
        return float(result1), float(result2)


NODE_CLASS_MAPPINGS = {
    "VWo50 Get Type": GetType,
    "VWo50 Get Element Type": GetElementType,
    "VWo50 Get Element At Index": GetElementAtIndex,
    "VWo50 Get Mask For IPAdapter Image": GetMaskForIPAdapterImage,
    "VWo50 Get Mask For Image Scale To Fit Rect": GetMaskForImageScaleToFitRect,
    "VWo50 Get Info of torch Tensor": GetInfoOfTorchTensor,
    "VWo50 Execute Python Code": ExecutePythonCode,
    "VWo50 Execute Python Code Return Int": ExecutePythonCodeReturnInt,
    "VWo50 Execute Python Code Return Float": ExecutePythonCodeReturnFloat,
}
