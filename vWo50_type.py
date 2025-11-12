class VWo50ToBool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为BOOL"

    def process(self, any_object):
        return (bool(any_object),)


class VWo50ToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为INT"

    def process(self, any_object):
        return (int(any_object),)


class VWo50ToFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为FLOAT"

    def process(self, any_object):
        return (float(any_object),)


class VWo50ToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为STRING"

    def process(self, any_object):
        return (str(any_object),)


class VWo50ToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为IMAGE"

    def process(self, any_object):
        return (any_object,)


class VWo50ToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为MASK"

    def process(self, any_object):
        return (any_object,)


class VWo50ToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为LATENT"

    def process(self, any_object):
        return (any_object,)


class VWo50ToVae:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("VAE",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为VAE"

    def process(self, any_object):
        return (any_object,)


class VWo50ToModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为MODEL"

    def process(self, any_object):
        return (any_object,)


class VWo50ToClip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为CLIP"

    def process(self, any_object):
        return (any_object,)


class VWo50ToConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为CONDITIONING"

    def process(self, any_object):
        return (any_object,)


class VWo50ToUpscaleModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_object": ("*",)},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "process"
    CATEGORY = "VWo50/Type"
    DESCRIPTION = "把类型转换为UPSCALE_MODEL"

    def process(self, any_object):
        return (any_object,)


NODE_CLASS_MAPPINGS = {
    "VWo50 To Bool": VWo50ToBool,
    "VWo50 To Int": VWo50ToInt,
    "VWo50 To Float": VWo50ToFloat,
    "VWo50 To String": VWo50ToString,
    "VWo50 To Image": VWo50ToImage,
    "VWo50 To MASK": VWo50ToMask,
    "VWo50 To Latent": VWo50ToLatent,
    "VWo50 To vae": VWo50ToVae,
    "VWo50 To Model": VWo50ToModel,
    "VWo50 To clip": VWo50ToClip,
    "VWo50 To Conditioning": VWo50ToConditioning,
    "VWo50 To Upscale Model": VWo50ToUpscaleModel,
}
