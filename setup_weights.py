"""
Download model weights to the weights/ directory.
Run once before using enhance.py.

    python setup_weights.py           # 下载全部
    python setup_weights.py --anime   # 只下载 anime 模型
"""

import argparse
import os
import urllib.request
from pathlib import Path

WEIGHTS_DEFAULT = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "GFPGANv1.3.pth":        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}

WEIGHTS_ANIME = {
    # Lighter 6-block model, better for AI-generated / illustration-style images
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}


def download(name: str, url: str, dest: Path):
    if dest.exists():
        print(f"  [SKIP] Already exists: {name}")
        return
    print(f"  [DOWN] {name} ...")

    def progress(count, block, total):
        pct = count * block / total * 100
        print(f"\r         {min(pct, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r  [OK]   {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anime", action="store_true",
                        help="Also download anime/illustration model")
    args = parser.parse_args()

    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    targets = dict(WEIGHTS_DEFAULT)
    if args.anime:
        targets.update(WEIGHTS_ANIME)

    print("Downloading model weights ...\n")
    for name, url in targets.items():
        download(name, url, weights_dir / name)

    print("\nAll weights ready. You can now run enhance.py.")


if __name__ == "__main__":
    main()
