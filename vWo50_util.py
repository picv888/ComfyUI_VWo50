from typing import Any, Mapping
import torch
import math


class VWo50GetType:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"any_object": ("*", {})}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "获取类型信息"

    def process(self, any_object: Any) -> tuple[str]:
        return (f"{type(any_object).__module__}.{type(any_object).__name__}",)


class VWo50GetElementType:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"container": ("*", {})}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "获取下标为0的元素的类型信息"

    def process(self, container: list) -> tuple[str]:
        any_object = container.index(0, None)
        return (f"{type(any_object).__module__}.{type(any_object).__name__}",)


class VWo50GetMaskForIPAdapterImage:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"image": ("IMAGE", {})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = (
        "使用IPAdapter时，图片都规定是正方形（一般1024x1024）的，当我们的图片是长方形时，可以以scale to "
        "fit的方式调整图片大小然后叠加在一个1024x1024的空白图片上，然后把实际图片内容作为蒙版传递给IPAdapter。这个节点就是用于获取这个蒙版"
    )

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
        mask[0, y : y + mask_height, x : x + mask_width] = 1.0

        return (mask,)


class VWo50GetMaskForImageScaleToFitRect:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "rect_width": ("INT", {}),
                "rect_height": ("INT", {}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "在一个宽为rectWidth，高为rectHeight的矩形框内，把一个图片以scale to fit的方式调整大小，获取这个图片矩形框的蒙版"

    def process(
        self, image: torch.Tensor, rect_width: int, rect_height: int
    ) -> tuple[torch.Tensor]:
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
        mask[0, y : y + mask_height, x : x + mask_width] = 1.0

        return (mask,)


class VWo50GetInfoOfTorchTensor:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"tensor_obj": ("*", {})}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "获取torch.Tensor对象的信息"

    def process(self, tensor_obj: Any) -> tuple[str]:
        if tensor_obj is None:
            return ("输入为空",)
        return (
            f"dimension: {str(tensor_obj.ndimension())}, shape: {str(tensor_obj.shape)}, dtype: {str(tensor_obj.dtype)}, device: {str(tensor_obj.device)}",
        )


class VWo50ExecutePythonCode:
    @classmethod
    def INPUT_TYPES(s) -> Mapping[str, Any]:
        return {
            "required": {},
            "optional": {
                "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "argument1": ("*", {}),
                "argument2": ("*", {}),
                "argument3": ("*", {}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "动态执行python代码"

    def process(
        self,
        code: str,
        argument1: Any = None,
        argument2: Any = None,
        argument3: Any = None,
    ) -> tuple[Any, Any]:
        result1 = ""
        result2 = ""
        local_vars = {
            "argument1": argument1,
            "argument2": argument2,
            "argument3": argument3,
        }
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", None)
        result2 = local_vars.get("result2", None)
        return result1, result2


class VWo50ExecutePythonCodeReturnBool:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {},
            "optional": {
                "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "argument1": ("*",),
                "argument2": ("*",),
                "argument3": ("*",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "动态执行python代码"

    def process(
        self,
        code: str,
        argument1: Any = None,
        argument2: Any = None,
        argument3: Any = None,
    ) -> tuple[bool, bool]:
        result1 = True
        result2 = True
        local_vars = {
            "argument1": argument1,
            "argument2": argument2,
            "argument3": argument3,
        }
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", False)
        result2 = local_vars.get("result2", False)
        return bool(result1), bool(result2)


class VWo50ExecutePythonCodeReturnInt:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {},
            "optional": {
                "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "argument1": ("*",),
                "argument2": ("*",),
                "argument3": ("*",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "动态执行python代码"

    def process(
        self,
        code: str,
        argument1: Any = None,
        argument2: Any = None,
        argument3: Any = None,
    ) -> tuple[int, int]:
        result1 = 0
        result2 = 0
        local_vars = {
            "argument1": argument1,
            "argument2": argument2,
            "argument3": argument3,
        }
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", 0)
        result2 = local_vars.get("result2", 0)
        return int(result1), int(result2)


class VWo50ExecutePythonCodeReturnFloat:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {},
            "optional": {
                "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "argument1": ("*",),
                "argument2": ("*",),
                "argument3": ("*",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "动态执行python代码"

    def process(
        self,
        code: str,
        argument1: Any = None,
        argument2: Any = None,
        argument3: Any = None,
    ) -> tuple[float, float]:
        result1 = 0.0
        result2 = 0.0
        local_vars = {
            "argument1": argument1,
            "argument2": argument2,
            "argument3": argument3,
        }
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", "nan")
        result2 = local_vars.get("result2", "nan")
        return float(result1), float(result2)


class VWo50ExecutePythonCodeReturnString:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {},
            "optional": {
                "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "argument1": ("*",),
                "argument2": ("*",),
                "argument3": ("*",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "动态执行python代码"

    def process(
        self,
        code: str,
        argument1: Any = None,
        argument2: Any = None,
        argument3: Any = None,
    ) -> tuple[float, float]:
        result1 = ""
        result2 = ""
        local_vars = {
            "argument1": argument1,
            "argument2": argument2,
            "argument3": argument3,
        }
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", "")
        result2 = local_vars.get("result2", "")
        return str(result1), str(result2)


import comfy
import torch


class VWo50GetNodeName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*", {}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any_input",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "获取上游节点的名字，这个名字和python代码中 NODE_CLASS_MAPPINGS 字典里的键一样，可以通过搜索这个名字快速找到该节点的代码"

    def process(self, any_input=None):
        # 这个节点的主要逻辑在前端，这里只是透传数据
        return (any_input,)


class VWo50SwitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("INT",),
                "input2": ("ANY",),
                "switchName": ("STRING", {"default": "Switch1"}),
            }
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_func"
    CATEGORY = "VWo50/Utils"

    def switch_func(self, input1, input2, switchName):
        self.case = input1
        self.obj = input2
        self.switchName = switchName
        return (input2,)


class VWo50CaseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("ANY",),
                "input2": ("ANY",),
                "switchName": ("STRING", {"default": "Switch1"}),
                "case": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("ANY", "ANY")
    RETURN_NAMES = ("output1", "output2")
    FUNCTION = "case_func"
    CATEGORY = "VWo50/Utils"

    def case_func(self, input1, input2, switchName, case):
        self.obj = input1
        self.obj2 = input2
        self.switchName = switchName
        self.case = case
        return (input1, input2)


NODE_CLASS_MAPPINGS = {
    "VWo50 Get Type": VWo50GetType,
    "VWo50 Get Element Type": VWo50GetElementType,
    "VWo50 Get Mask For IPAdapter Image": VWo50GetMaskForIPAdapterImage,
    "VWo50 Get Mask For Image Scale To Fit Rect": VWo50GetMaskForImageScaleToFitRect,
    "VWo50 Get Info of torch Tensor": VWo50GetInfoOfTorchTensor,
    "VWo50 Execute Python Code": VWo50ExecutePythonCode,
    "VWo50 Execute Python Code Return Bool": VWo50ExecutePythonCodeReturnBool,
    "VWo50 Execute Python Code Return Int": VWo50ExecutePythonCodeReturnInt,
    "VWo50 Execute Python Code Return Float": VWo50ExecutePythonCodeReturnFloat,
    "VWo50 Execute Python Code Return String": VWo50ExecutePythonCodeReturnString,
    "VWo50 Get Node Name": VWo50GetNodeName,
    "VWo50 SwitchNode": VWo50SwitchNode,
    "VWo50 CaseNode": VWo50CaseNode,
}
