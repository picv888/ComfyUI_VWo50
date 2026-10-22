import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import tempfile


class VWo50PreviewImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "mask_file": ("STRING", {"default": ""})
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "VWo50/image"
    OUTPUT_NODE = False
    HAS_INTERMEDIATE_OUTPUT = True  # 使命中缓存时，前端的 nodeType.prototype.onExecuted 仍然被调用

    @classmethod
    def IS_CHANGED(cls, image, mask_file=""):
        return mask_file

    def process(self, image, mask_file=""):
        # 透传原图张量
        output_image = image
        # 预览文件（关键：供前端显示 & MaskEditor 使用） ---
        preview_images = []
        # 读取遮罩
        mask_tensor = None
        if mask_file:
            try:
                mask_path = folder_paths.get_annotated_filepath(mask_file)
                if os.path.exists(mask_path):
                    img = Image.open(mask_path)
                    # 判断 mask_file 是否属于当前 image
                    if not self._mask_matches_image(img, image):
                        mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
                        preview_images.append(self._tensor_to_temp_png(image))
                    else:
                        if img.mode in ("RGBA", "LA"):
                            # 带 alpha 的图：取 alpha
                            a = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                        elif img.mode == "P" and "transparency" in img.info:
                            # 调色板 + 透明色
                            img = img.convert("RGBA")
                            a = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                        else:
                            # 没有 alpha，退化为用灰度/二值作为 mask
                            a = np.array(img.convert("L")).astype(np.float32) / 255.0
                        mask_tensor = 1.0 - torch.from_numpy(a)
                        mask_tensor = mask_tensor.unsqueeze(0)
                        preview_images.append(
                            {"filename": mask_file, "subfolder": "", "type": "input"}
                        )

            except Exception as e:
                print("发生了异常:", e)
                # 64x64 mask
                mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
                # 原图预览
                preview_images.append(self._tensor_to_temp_png(image))

        else:  # 没有mask
            # 返回 64x64 的空mask
            mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
            # 原图预览
            preview_images.append(self._tensor_to_temp_png(image))

        # 构造 ui 返回值
        ui = {}
        if preview_images:
            # 前端 onExecuted 会读取 ui.images
            # 格式参考官方 PreviewImage 节点的 ui 输出
            ui["images"] = preview_images

        return {"ui": ui, "result": (output_image, mask_tensor)}

    def _mask_matches_image(
        self, mask_img: Image.Image, image_tensor, tol: float = 0.02
    ) -> bool:
        """
        判断 mask 文件是否对应当前 image。
        MaskEditor 保存的 PNG：RGB=原图，A=mask。
        只要 RGB 内容一致，就认为 mask 有效。
        """
        try:
            # 1) 当前 image -> numpy [H, W, 3]，float32 [0,1]
            if isinstance(image_tensor, torch.Tensor):
                img_np = image_tensor
                if img_np.ndim == 4:
                    img_np = img_np[0]
                img_np = img_np[..., :3].float().cpu().numpy()
            else:
                img_np = np.asarray(image_tensor)[..., :3].astype(np.float32)

            H, W = img_np.shape[:2]

            # 2) 尺寸必须一致（MaskEditor 不缩放，尺寸变了肯定不是同一张）
            if mask_img.size != (W, H):  # PIL size = (W, H)
                return False

            # 3) 比对 RGB 内容
            mask_rgb = np.array(mask_img.convert("RGB")).astype(np.float32) / 255.0
            if mask_rgb.shape != img_np.shape:
                return False

            diff = float(np.abs(mask_rgb - img_np).mean())
            print(f"[mask_check] RGB mean diff = {diff:.5f}")
            return diff < tol
        except Exception as e:
            print(f"[mask_check] 比对异常: {e}")
            return False

    def _tensor_to_temp_png(self, tensor):
        """将 [B,H,W,C] 张量转换为临时 PNG 文件信息"""
        # 取批次第一张，并转为 numpy
        arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        return self._array_to_temp_png(arr)

    def _array_to_temp_png(self, arr):
        """将 numpy 数组保存为临时 PNG，返回前端可用的文件信息字典"""
        img = Image.fromarray(arr)
        # 保存到 ComfyUI 的 temp 目录
        temp_dir = folder_paths.get_temp_directory()
        fd, path = tempfile.mkstemp(suffix=".png", dir=temp_dir)
        os.close(fd)
        img.save(path)
        # 返回与官方节点兼容的引用格式
        filename = os.path.basename(path)
        return {"filename": filename, "subfolder": "", "type": "temp"}


NODE_CLASS_MAPPINGS = {
    "VWo50 Preview Image": VWo50PreviewImage,
}
