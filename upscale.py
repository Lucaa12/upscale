"""ParadoxAI Upscale: the upscale box of the ParadoxAI video workflows (the ParadoxAI Upscale Pass).

A fast replacement for ComfyUI's "Upscale Image (using Model)". ComfyUI's own box blends tiles on the CPU, upscales one
frame at a time and always tiles at 512. This box keeps the work on the graphics card, upscales several frames at once,
builds each tile's blend mask once, and does not tile by default (tiling is slower and loses detail at the seams).
If the card runs out of memory it halves the batch, then tiles, and tries again.

It calls the upscale model through spandrel's wrapper (`upscale_model(x)`), never the raw `.model`: the wrapper pads the
frame to the size the model needs and removes the padding afterwards.

Based on MIT-licensed code (see LICENSE in this folder).
"""

import logging

import torch

import comfy.model_management as mm
import comfy.utils

logger = logging.getLogger(__name__)

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

_OOM = getattr(mm, "OOM_EXCEPTION", torch.cuda.OutOfMemoryError)

# A flat per-input-pixel activation budget instead of stock's
# `(512*512*3)*elem*scale*384` fiction. It over-estimates small archs and may
# under-estimate heavy ones -- the OOM backoff in `upscale()` is the safety net,
# so a wrong guess costs one retry, not a crash.
_ACT_BYTES_PER_PIXEL = 96


def _pick_dtype(upscale_model, precision, device):
    """Resolve `precision` against what the arch and the device actually support.

    `auto` picks fp32 on CUDA, which is not the obvious answer. Measured on an H200 with
    SPAN x2 (16 frames @ 720p): fp32 was both the *fastest* and the most accurate config,
    beating every fp16/bf16 variant. `torch.backends.cudnn.allow_tf32` defaults to True, so
    fp32 convolutions already run on tensor cores at TF32 rate; these upscalers are small
    and bandwidth-bound, so halving the bytes does not pay for the cast overhead.
    Low precision here buys VRAM headroom, not speed -- so it stays a manual knob.
    """
    if precision != "auto":
        want = _DTYPES[precision]
    elif device.type == "mps":
        want = torch.float16  # no TF32 on Metal, so fp16 does help there
    else:
        want = torch.float32

    # spandrel exposes these per-architecture; stock never reads them.
    if want is torch.bfloat16 and not getattr(upscale_model, "supports_bfloat16", False):
        want = torch.float16 if getattr(upscale_model, "supports_half", False) else torch.float32
    if want is torch.float16 and not getattr(upscale_model, "supports_half", False):
        want = torch.float32

    if precision != "auto" and want is not _DTYPES[precision]:
        logger.warning(
            "ParadoxAI Upscale: %s does not support %s, falling back to %s",
            getattr(upscale_model, "architecture", "model"), precision, want,
        )
    return want


def _feather_mask(h, w, feather_y, feather_x, device, dtype):
    """Separable linear edge ramp, the same weighting as `comfy.utils.tiled_scale`, built once."""
    def ramp(n, feather):
        m = torch.ones(n, device=device, dtype=dtype)
        if 0 < feather < n:
            r = (torch.arange(feather, device=device, dtype=dtype) + 1) / feather
            m[:feather] *= r
            m[n - feather:] *= r.flip(0)
        return m

    return (ramp(h, feather_y)[:, None] * ramp(w, feather_x)[None, :]).view(1, 1, h, w)


class ParadoxAI_Upscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "auto is the fastest and sharpest. Pick fp16 or bf16 only if the "
                               "graphics card runs out of memory: they use less, but are slower.",
                }),
                "batch_size": ("INT", {
                    "default": 8, "min": 1, "max": 512,
                    "tooltip": "How many frames are upscaled at once. More uses more graphics "
                               "memory. If the card runs out, the box halves it by itself.",
                }),
                "tile_size": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "0 = the whole frame in one go (best). A number cuts each frame "
                               "into squares of that size: slower and a little softer at the "
                               "joins, only for cards that cannot fit a whole frame.",
                }),
                "overlap": ("INT", {
                    "default": 32, "min": 0, "max": 256, "step": 8,
                    "tooltip": "How much the squares overlap. Only used when tile_size is above 0.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "ParadoxAI"
    DESCRIPTION = ("ParadoxAI Upscale: makes every frame bigger and sharper with an upscale "
                   "model, several frames at once on the graphics card.")

    def upscale(self, upscale_model, image, precision, batch_size, tile_size, overlap):
        device = mm.get_torch_device()
        dtype = _pick_dtype(upscale_model, precision, device)
        scale = upscale_model.scale

        in_img = image.movedim(-1, -3)
        n, _, h, w = in_img.shape
        oh, ow = round(h * scale), round(w * scale)

        self._load(upscale_model, device, dtype, min(batch_size, n), h, w, scale)

        out_device = mm.intermediate_device()
        out = torch.empty((n, 3, oh, ow), device=out_device, dtype=torch.float32)

        pbar = comfy.utils.ProgressBar(n)
        i = 0
        while i < n:
            count = min(batch_size, n - i)
            chunk = result = None
            try:
                chunk = in_img[i:i + count].to(device=device, dtype=dtype)
                if device.type == "cuda":
                    chunk = chunk.contiguous(memory_format=torch.channels_last)
                result = self._run(upscale_model, chunk, tile_size, overlap, scale)
            except _OOM:
                chunk = result = None
                mm.soft_empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.warning("ParadoxAI Upscale: OOM, retrying at batch_size=%d", batch_size)
                    continue
                if tile_size == 0 or tile_size > 128:
                    tile_size = 512 if tile_size == 0 else tile_size // 2
                    logger.warning("ParadoxAI Upscale: OOM, retrying at tile_size=%d", tile_size)
                    continue
                raise

            out[i:i + count] = result.to(device=out_device, dtype=torch.float32)
            del chunk, result
            pbar.update(count)
            i += count

        return (out.movedim(-3, -1).clamp_(0.0, 1.0),)

    def _load(self, upscale_model, device, dtype, n, h, w, scale):
        """Get the model resident at the right device/dtype, via ComfyUI's memory manager."""
        patcher = getattr(upscale_model, "patcher", None)
        if patcher is not None:
            elem = torch.tensor([], dtype=dtype).element_size()
            required = n * h * w * _ACT_BYTES_PER_PIXEL * elem          # activations
            required += n * 3 * round(h * scale) * round(w * scale) * 4  # fp32 blend buffer
            try:
                mm.load_models_gpu([patcher], memory_required=int(required))
            except Exception as e:  # noqa: BLE001 - never let bookkeeping break the run
                logger.warning("ParadoxAI Upscale: load_models_gpu failed (%s), loading directly", e)

        if upscale_model.device != device or upscale_model.dtype != dtype:
            upscale_model.to(device=device, dtype=dtype)
        if device.type == "cuda":
            upscale_model.model.to(memory_format=torch.channels_last)

    def _run(self, model, chunk, tile_size, overlap, scale):
        _, _, h, w = chunk.shape
        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            return model(chunk)
        return self._tiled(model, chunk, tile_size, overlap, scale)

    def _tiled(self, model, chunk, tile, overlap, scale):
        b, _, h, w = chunk.shape
        oh, ow = round(h * scale), round(w * scale)
        device = chunk.device

        acc = torch.zeros((b, 3, oh, ow), device=device, dtype=torch.float32)
        weight = torch.zeros((1, 1, oh, ow), device=device, dtype=torch.float32)
        feather = round(overlap * scale)
        masks = {}

        step = max(1, tile - overlap)
        ys = range(0, max(1, h - overlap), step)
        xs = range(0, max(1, w - overlap), step)

        for y in ys:
            for x in xs:
                y0 = max(0, min(h - overlap, y))
                x0 = max(0, min(w - overlap, x))
                piece = model(chunk[:, :, y0:y0 + tile, x0:x0 + tile]).float()

                ph, pw = piece.shape[2], piece.shape[3]
                if (ph, pw) not in masks:
                    masks[(ph, pw)] = _feather_mask(ph, pw, feather, feather, device, torch.float32)
                mask = masks[(ph, pw)]

                oy, ox = round(y0 * scale), round(x0 * scale)
                ch, cw = min(ph, oh - oy), min(pw, ow - ox)
                acc[:, :, oy:oy + ch, ox:ox + cw] += piece[:, :, :ch, :cw] * mask[:, :, :ch, :cw]
                weight[:, :, oy:oy + ch, ox:ox + cw] += mask[:, :, :ch, :cw]

        return acc.div_(weight.clamp_min_(1e-8))
