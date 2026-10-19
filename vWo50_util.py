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
    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("result1", "result2")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "动态执行python代码"

    @classmethod
    def INPUT_TYPES(s) -> Mapping[str, Any]:
        return {
            "required": {},
            "optional": {
                "code": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "input_0": ("*", {}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(
        self,
        code: str,
        **kwargs,
    ):
        # 收集所有 input
        inputs: dict = {k: v for k, v in kwargs.items() if k.startswith("input_")}
        result1 = ""
        result2 = ""
        local_vars = {
            **inputs,
        }
        exec(code, {}, local_vars)
        result1 = local_vars.get("result1", None)
        result2 = local_vars.get("result2", None)
        return (result1, result2)


class VWo50MakeListAndGetElementAtIndex:
    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("list", "element")
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "把输入的所有参数组合成一个List，根据index获取元素"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"index": ("INT", {"default": 0, "min": 0, "max": 999999})},
            "optional": {
                # 格式："input_{i}": (类型, 默认)
                "input_0": ("*", {}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(
        self,
        index,
        **kwargs,
    ):
        input_items = sorted(
            [(k, v) for k, v in kwargs.items() if k.startswith("input_")],
            key=lambda kv: int(kv[0].split("_", 1)[1]),
        )
        inputs = [v for _, v in input_items]
        print(f"list有一共有{len(inputs)}个元素，你选择了第{index}个（从下标0开始）")
        element = inputs[index] if index < len(inputs) else None
        return (inputs, element)


class VWo50Switch:
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Utils"
    DESCRIPTION = "选择一个上游分支执行，其它分支都不会执行"
    

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # 预先声明足够多的 lazy 输入（数量可按需调整）
        for i in range(50):
            optional[f"input_{i}"] = ("*", {"lazy": True})
        return {
            "required": {"index": ("INT", {"default": 0, "min": 0, "max": 999999})},
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    # 根据 index 决定需要求值哪个输入
    def check_lazy_status(
        self,
        index,
        **kwargs,
    ):
        target_input = f"input_{index}"
        # 如果目标输入尚未被求值（为 None），则请求求值它
        if kwargs.get(target_input) is None:
            return [target_input]
        # 否则，不需要再求值任何其他 lazy 输入
        return []

    def process(
        self,
        index,
        **kwargs,
    ):
        target = f"input_{index}"
        element = kwargs.get(target, None)
        print(f"VWo50Switch节点: 选择 {target}")
        return (element,)


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


NODE_CLASS_MAPPINGS = {
    "VWo50 Get Type": VWo50GetType,
    "VWo50 Get Element Type": VWo50GetElementType,
    "VWo50 Get Mask For IPAdapter Image": VWo50GetMaskForIPAdapterImage,
    "VWo50 Get Mask For Image Scale To Fit Rect": VWo50GetMaskForImageScaleToFitRect,
    "VWo50 Get Info of torch Tensor": VWo50GetInfoOfTorchTensor,
    "VWo50 Execute Python Code": VWo50ExecutePythonCode,
    "VWo50 Make List And Get Element At Index": VWo50MakeListAndGetElementAtIndex,
    "VWo50 Switch": VWo50Switch,
    "VWo50 Get Node Name": VWo50GetNodeName,
}
