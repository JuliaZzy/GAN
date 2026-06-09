"""
Image enhancement using Real-ESRGAN + GFPGAN.
Auto-detects device: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.

Usage:
    Single image:  python enhance.py -i blurry.jpg -o sharp.jpg
    Batch folder:  python enhance.py -i input/ -o output/
    Face mode:     python enhance.py -i blurry.jpg -o sharp.jpg --face
    2x upscale:    python enhance.py -i blurry.jpg -o sharp.jpg -s 2
    16x upscale:   python enhance.py -i blurry.jpg -o sharp.jpg --outscale 16
    Portrait 16x:  python enhance.py -i blurry.jpg -o sharp.jpg --face --outscale 16
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

WEIGHT_ESRGAN_X4       = "weights/RealESRGAN_x4plus.pth"
WEIGHT_ESRGAN_X2       = "weights/RealESRGAN_x2plus.pth"
WEIGHT_ESRGAN_ANIME_X4 = "weights/RealESRGAN_x4plus_anime_6B.pth"
WEIGHT_GFPGAN          = "weights/GFPGANv1.3.pth"


def detect_device() -> tuple[str, bool, str]:
    """Returns (device, use_half, label) for the best available accelerator."""
    import torch
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda", True, f"{vram:.1f} GB VRAM"
    if torch.backends.mps.is_available():
        return "mps", False, "Apple Silicon (unified memory)"
    return "cpu", False, "CPU only"


def auto_tile(device: str) -> int:
    import torch
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 16:
            return 0    # 3090/4090: no tiling
        elif vram >= 10:
            return 512
        else:
            return 256  # 4060 8GB etc.
    if device == "mps":
        return 512      # unified memory, conservative default
    return 128          # CPU


def max_safe_outscale(h: int, w: int, device: str) -> int:
    """Heuristic cap to avoid OOM during chained upscaling passes."""
    import torch
    pixels = h * w
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 16 and pixels <= 2_000_000:
            return 64
        if vram >= 10 and pixels <= 1_000_000:
            return 32
        if pixels <= 1_500_000:
            return 16
        return 8
    if device == "mps":
        return 16 if pixels <= 1_000_000 else 8
    return 8


def release_vram():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_esrgan(scale: int, anime: bool = False):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if anime:
        if scale != 4:
            print("[WARN] Anime model only supports 4x, ignoring -s flag.")
            scale = 4
        weight    = WEIGHT_ESRGAN_ANIME_X4
        num_block = 6   # lighter 6-block architecture
    else:
        weight    = WEIGHT_ESRGAN_X4 if scale == 4 else WEIGHT_ESRGAN_X2
        num_block = 23

    if not os.path.exists(weight):
        hint = "python setup_weights.py --anime" if anime else "python setup_weights.py"
        sys.exit(f"[ERROR] Weight file not found: {weight}\nRun: {hint}")

    device, use_half, label = detect_device()
    tile = auto_tile(device)
    model_tag = "anime-6B" if anime else f"x{scale}plus"
    print(f"  Model: {model_tag}  Device: {device} ({label})  tile={'off' if tile == 0 else tile}  fp16={use_half}")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=num_block, num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=weight,
        model=model,
        tile=tile,
        tile_pad=10,
        half=use_half,
        device=device,
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


def enhance_pass(img: np.ndarray, upsampler, face_restorer, model_scale: int, use_face: bool) -> np.ndarray:
    if use_face and face_restorer is not None:
        _, _, output = face_restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        return output
    output, _ = upsampler.enhance(img, outscale=model_scale)
    return output


def enhance_single(
    img: np.ndarray,
    upsampler,
    face_restorer,
    model_scale: int,
    outscale: int,
    use_face: bool,
) -> np.ndarray:
    if outscale <= model_scale:
        if use_face and face_restorer is not None:
            _, _, output = face_restorer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
            if outscale != model_scale:
                h, w = img.shape[:2]
                output = cv2.resize(
                    output,
                    (int(w * outscale), int(h * outscale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            return output
        output, _ = upsampler.enhance(img, outscale=outscale)
        return output

    current = img
    factor = 1
    pass_idx = 0
    while factor < outscale:
        if factor * model_scale > outscale:
            remaining = outscale / factor
            print(f"    pass {pass_idx + 1}: {factor}x -> {outscale}x (final resize)")
            if use_face and pass_idx == 0 and face_restorer is not None:
                _, _, current = face_restorer.enhance(
                    current, has_aligned=False, only_center_face=False, paste_back=True
                )
                if remaining != model_scale:
                    h, w = img.shape[:2]
                    current = cv2.resize(
                        current,
                        (int(w * outscale), int(h * outscale)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                return current
            output, _ = upsampler.enhance(current, outscale=remaining)
            return output

        pass_idx += 1
        face_this_pass = use_face and pass_idx == 1
        print(f"    pass {pass_idx}: {factor}x -> {factor * model_scale}x"
              + (" (GFPGAN face restore)" if face_this_pass else ""))
        current = enhance_pass(current, upsampler, face_restorer, model_scale, face_this_pass)
        factor *= model_scale
        release_vram()

    return current


def process_file(
    src: Path,
    dst: Path,
    upsampler,
    face_restorer,
    model_scale: int,
    outscale: int,
    use_face: bool,
):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [SKIP] Cannot read: {src}")
        return

    h, w = img.shape[:2]
    print(f"  {src.name}: {w}x{h} -> target {w * outscale}x{h * outscale} ({outscale}x)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    output = enhance_single(img, upsampler, face_restorer, model_scale, outscale, use_face)

    # Always save as PNG to avoid JPEG re-compression artefacts
    out_path = dst.with_suffix(".png")
    cv2.imwrite(str(out_path), output)
    print(f"  [OK] {src.name} -> {out_path.name} ({output.shape[1]}x{output.shape[0]})")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Enhance blurry images with Real-ESRGAN / GFPGAN")
    parser.add_argument("-i", "--input",  required=True, help="Input image or folder")
    parser.add_argument("-o", "--output", required=True, help="Output image or folder")
    parser.add_argument("-s", "--scale", type=int, default=4, choices=[2, 4],
                        help="Model upscale factor per pass (default: 4)")
    parser.add_argument("--outscale", type=int, default=None,
                        help="Final target upscale factor, e.g. 16 for 4x4 chained passes (default: same as -s)")
    parser.add_argument("--face", action="store_true",
                        help="Enable GFPGAN face restoration (recommended for portraits)")
    parser.add_argument("--anime", action="store_true",
                        help="Use anime/illustration model (less noise, better for AI-generated art)")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    model_scale = args.scale
    outscale = args.outscale if args.outscale is not None else model_scale

    if outscale < 1:
        sys.exit("[ERROR] --outscale must be >= 1")
    if outscale > 100:
        print("[WARN] 100x+ upscale is not practical on consumer GPUs and will not recover lost detail.")
        print("       AI super-resolution invents texture; extreme scales mostly enlarge blur.")

    images = collect_images(src)
    if not images:
        sys.exit(f"[ERROR] No supported images found in: {src}")

    device, _, _ = detect_device()
    sample = cv2.imread(str(images[0]))
    if sample is not None:
        sh, sw = sample.shape[:2]
        cap = max_safe_outscale(sh, sw, device)
        if outscale > cap:
            print(f"[WARN] Requested {outscale}x exceeds safe limit ~{cap}x for {sw}x{sh} on this device.")
            print(f"       Capping to {cap}x to avoid GPU out-of-memory.")
            outscale = cap

    print(f"Found {len(images)} image(s). Building model "
          f"(model={model_scale}x/pass, target={outscale}x, face={args.face}) ...")

    upsampler = build_esrgan(model_scale, anime=args.anime)
    face_restorer = build_gfpgan(model_scale, upsampler) if args.face else None

    print("Model loaded. Processing ...\n")
    for img_path in images:
        # Preserve relative sub-folder structure when input is a directory
        rel = img_path.relative_to(src) if src.is_dir() else Path(img_path.name)
        out_path = (dst / rel) if src.is_dir() else dst
        process_file(img_path, out_path, upsampler, face_restorer, model_scale, outscale, args.face)

    print("\nDone.")


if __name__ == "__main__":
    main()
