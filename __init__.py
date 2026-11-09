"""ParadoxAI Upscale, the upscale box of the ParadoxAI video workflows."""
from .upscale import ParadoxAI_Upscale


class ParadoxAI_Upscale_OldFiles(ParadoxAI_Upscale):
    """The same box under the type name that workflow files saved before 25 October 2026 use, so those files still
    open and run. Hidden from the box search (DEPRECATED), and shown under the ParadoxAI name."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "ParadoxAI-Upscale": ParadoxAI_Upscale,
    "HearmemanAI-UpscaleModel": ParadoxAI_Upscale_OldFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParadoxAI-Upscale": "ParadoxAI Upscale",
    "HearmemanAI-UpscaleModel": "ParadoxAI Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
