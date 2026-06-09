"""
SDXL + ControlNet Tile upscaler — optimised for RTX 3090 (24 GB).

How it works:
  1. Lanczos pre-upscale to target resolution
  2. Slice into overlapping 1024×1024 tiles (SDXL native resolution)
  3. Run SDXL img2img on every tile, guided by ControlNet Tile
  4. Feather-blend tiles back into one seamless image

Models are auto-downloaded from HuggingFace on first run (~10 GB total).
They are cached in HF_HOME (default: ~/.cache/huggingface).

Usage:
    # Basic (2x, auto-prompt)
    python upscale_sd.py -i input/palace.jpg -o output/palace_hd.png

    # Describe the image for better detail generation
    python upscale_sd.py -i input/palace.jpg -o output/palace_hd.png \\
        -p "Chinese imperial palace, night, red lanterns, flowing robes, 8k"

    # 4x upscale, slightly more aggressive redraw
    python upscale_sd.py -i input/palace.jpg -o output/palace_4x.png -s 4 --denoise 0.45
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# HuggingFace model IDs (auto-downloaded on first run)
SDXL_MODEL       = "stabilityai/stable-diffusion-xl-base-1.0"   # ~7 GB
CONTROLNET_MODEL = "xinsir/controlnet-tile-sdxl-1.0"            # ~2.5 GB

TILE_SIZE   = 1024   # SDXL native resolution
TILE_STRIDE = 768    # step between tiles → 256px overlap


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_pipeline():
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

    print("[1/2] Loading ControlNet Tile ...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16
    )
    print("[2/2] Loading SDXL ...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        SDXL_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # 3090 has plenty of VRAM — no CPU offload needed.
    # VAE slicing lets us decode very large tiles without OOM.
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# Tile blending
# ---------------------------------------------------------------------------

def feather_mask(h: int, w: int, overlap: int) -> np.ndarray:
    """Linear ramp at tile edges for seamless blending."""
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(overlap):
        v = (i + 1) / (overlap + 1)
        mask[i, :]    = np.minimum(mask[i, :],    v)
        mask[-(i+1), :] = np.minimum(mask[-(i+1), :], v)
        mask[:, i]    = np.minimum(mask[:, i],    v)
        mask[:, -(i+1)] = np.minimum(mask[:, -(i+1)], v)
    return mask


def tile_upscale(
    img: Image.Image,
    pipe,
    prompt: str,
    denoise: float,
    steps: int,
) -> Image.Image:
    overlap = TILE_SIZE - TILE_STRIDE
    w, h    = img.size

    # Pad so tiles cover the full image
    def pad_dim(d):
        if d <= TILE_SIZE:
            return TILE_SIZE
        return math.ceil((d - TILE_SIZE) / TILE_STRIDE) * TILE_STRIDE + TILE_SIZE

    pad_w, pad_h = pad_dim(w), pad_dim(h)
    canvas = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
    canvas.paste(img, (0, 0))

    out_arr    = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    weight_arr = np.zeros((pad_h, pad_w),    dtype=np.float32)

    ys = list(range(0, pad_h - TILE_SIZE + 1, TILE_STRIDE))
    xs = list(range(0, pad_w - TILE_SIZE + 1, TILE_STRIDE))
    total = len(ys) * len(xs)

    neg = "blurry, jpeg artifacts, low quality, lowres, watermark"

    for idx, (y, x) in enumerate([(y, x) for y in ys for x in xs], 1):
        print(f"  Tile {idx:>3}/{total}  ({x:4},{y:4})", end="\r", flush=True)

        tile = canvas.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            image=tile,
            control_image=tile,
            strength=denoise,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=steps,
        ).images[0]

        mask = feather_mask(TILE_SIZE, TILE_SIZE, overlap)
        out_arr[y:y+TILE_SIZE, x:x+TILE_SIZE]    += np.array(result, dtype=np.float32) * mask[:, :, None]
        weight_arr[y:y+TILE_SIZE, x:x+TILE_SIZE] += mask

    print(f"  {total}/{total} tiles done.     ")

    weight_arr = np.maximum(weight_arr, 1e-6)[:, :, None]
    blended = np.clip(out_arr / weight_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(blended).crop((0, 0, w, h))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SDXL + ControlNet Tile upscaler (3090 optimised)"
    )
    parser.add_argument("-i", "--input",   required=True, help="Input image path")
    parser.add_argument("-o", "--output",  required=True, help="Output image path")
    parser.add_argument(
        "-p", "--prompt",
        default="masterpiece, best quality, highly detailed, sharp focus, 8k",
        help="Describe image content — more specific = better detail (default: generic quality prompt)",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2, choices=[2, 4],
        help="Upscale factor (default: 2). 4x is ~4× slower.",
    )
    parser.add_argument(
        "--denoise", type=float, default=0.35,
        help="Redraw strength 0–1 (default 0.35). "
             "Lower = preserves original more; higher = more new detail but may drift.",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Inference steps per tile (default: 30). More steps = slower but slightly sharper.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output).with_suffix(".png")
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src).convert("RGB")
    ow, oh = img.size
    print(f"Input : {ow}×{oh}  →  target {ow*args.scale}×{oh*args.scale}  (scale={args.scale}x)")
    print(f"Prompt: {args.prompt}")
    print(f"Denoise: {args.denoise}  Steps: {args.steps}\n")

    # Step 1: Lanczos pre-upscale
    if args.scale > 1:
        img = img.resize((ow * args.scale, oh * args.scale), Image.LANCZOS)
        print(f"Lanczos pre-upscale done: {img.size[0]}×{img.size[1]}\n")

    # Step 2: SDXL tile pass
    pipe = load_pipeline()
    print("\nRunning SDXL tile pass ...")
    result = tile_upscale(img, pipe, args.prompt, args.denoise, args.steps)

    result.save(str(dst))
    print(f"\nSaved → {dst}  ({result.size[0]}×{result.size[1]})")


if __name__ == "__main__":
    main()
