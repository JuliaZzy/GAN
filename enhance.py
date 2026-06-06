"""
Image enhancement using Real-ESRGAN + GFPGAN.
GPU: RTX 3090 (24GB) — runs without tiling, fp16 enabled.

Usage:
    Single image:  python enhance.py -i blurry.jpg -o sharp.jpg
    Batch folder:  python enhance.py -i input/ -o output/
    Face mode:     python enhance.py -i blurry.jpg -o sharp.jpg --face
    2x upscale:    python enhance.py -i blurry.jpg -o sharp.jpg -s 2
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

WEIGHT_ESRGAN_X4 = "weights/RealESRGAN_x4plus.pth"
WEIGHT_ESRGAN_X2 = "weights/RealESRGAN_x2plus.pth"
WEIGHT_GFPGAN    = "weights/GFPGANv1.3.pth"


def build_esrgan(scale: int):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    weight = WEIGHT_ESRGAN_X4 if scale == 4 else WEIGHT_ESRGAN_X2
    if not os.path.exists(weight):
        sys.exit(
            f"[ERROR] Weight file not found: {weight}\n"
            "Run setup_weights.py to download model weights."
        )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=weight,
        model=model,
        tile=0,        # 3090 has 24GB, no tiling needed
        tile_pad=10,
        half=True,     # fp16: ~2x speed, negligible quality loss
        device="cuda",
    )
    return upsampler


def build_gfpgan(scale: int, bg_upsampler):
    from gfpgan import GFPGANer

    if not os.path.exists(WEIGHT_GFPGAN):
        sys.exit(
            f"[ERROR] Weight file not found: {WEIGHT_GFPGAN}\n"
            "Run setup_weights.py to download model weights."
        )

    restorer = GFPGANer(
        model_path=WEIGHT_GFPGAN,
        upscale=scale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
    )
    return restorer


def enhance_single(img: np.ndarray, upsampler, face_restorer, scale: int) -> np.ndarray:
    if face_restorer is not None:
        _, _, output = face_restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        return output
    else:
        output, _ = upsampler.enhance(img, outscale=scale)
        return output


def process_file(src: Path, dst: Path, upsampler, face_restorer, scale: int):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [SKIP] Cannot read: {src}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    output = enhance_single(img, upsampler, face_restorer, scale)

    # Always save as PNG to avoid JPEG re-compression artefacts
    out_path = dst.with_suffix(".png")
    cv2.imwrite(str(out_path), output)
    print(f"  [OK] {src.name} -> {out_path.name}")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Enhance blurry images with Real-ESRGAN / GFPGAN")
    parser.add_argument("-i", "--input",  required=True, help="Input image or folder")
    parser.add_argument("-o", "--output", required=True, help="Output image or folder")
    parser.add_argument("-s", "--scale",  type=int, default=4, choices=[2, 4], help="Upscale factor (default: 4)")
    parser.add_argument("--face", action="store_true", help="Enable GFPGAN face restoration")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    images = collect_images(src)
    if not images:
        sys.exit(f"[ERROR] No supported images found in: {src}")

    print(f"Found {len(images)} image(s). Building model (scale={args.scale}x, face={args.face}) ...")

    upsampler = build_esrgan(args.scale)
    face_restorer = build_gfpgan(args.scale, upsampler) if args.face else None

    print("Model loaded. Processing ...\n")
    for img_path in images:
        # Preserve relative sub-folder structure when input is a directory
        rel = img_path.relative_to(src) if src.is_dir() else Path(img_path.name)
        out_path = (dst / rel) if src.is_dir() else dst
        process_file(img_path, out_path, upsampler, face_restorer, args.scale)

    print("\nDone.")


if __name__ == "__main__":
    main()
