"""
Download model weights to the weights/ directory.
Run once before using enhance.py.
"""

import os
import urllib.request
from pathlib import Path

WEIGHTS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "GFPGANv1.3.pth":        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
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
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    print("Downloading model weights ...\n")
    for name, url in WEIGHTS.items():
        download(name, url, weights_dir / name)

    print("\nAll weights ready. You can now run enhance.py.")


if __name__ == "__main__":
    main()
