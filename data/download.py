"""
Download the CWRU Bearing Dataset (12kHz Drive End).

The Case Western Reserve University Bearing Data Center provides
accelerometer data collected at 12,000 samples/second from bearings
with seeded faults under various conditions.

Reference: https://engineering.case.edu/bearingdatacenter
"""

import os
import requests
from tqdm import tqdm

# CWRU dataset file mappings: fault_type -> (filename, url_key)
# These are the 12k Drive End accelerometer recordings at 1797 RPM (approx.)
CWRU_FILES = {
    "normal": {
        "url": "https://engineering.case.edu/sites/default/files/97.mat",
        "filename": "97.mat",
        "description": "Normal baseline, 0 HP load",
    },
    "IR007": {
        "url": "https://engineering.case.edu/sites/default/files/105.mat",
        "filename": "105.mat",
        "description": "Inner race fault 0.007 inch, 0 HP load",
    },
    "B007": {
        "url": "https://engineering.case.edu/sites/default/files/118.mat",
        "filename": "118.mat",
        "description": "Ball fault 0.007 inch, 0 HP load",
    },
    "OR007": {
        "url": "https://engineering.case.edu/sites/default/files/130.mat",
        "filename": "130.mat",
        "description": "Outer race fault 0.007 inch (centered), 0 HP load",
    },
}


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_cwru_dataset(output_dir: str = "data/raw") -> None:
    """Download all configured CWRU bearing fault .mat files."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading CWRU 12k Drive End dataset to: {output_dir}")
    print("=" * 60)

    for fault_type, info in CWRU_FILES.items():
        dest = os.path.join(output_dir, info["filename"])
        if os.path.exists(dest):
            print(f"[SKIP] {fault_type} ({info['filename']}) already exists.")
            continue

        print(f"[DOWNLOAD] {fault_type}: {info['description']}")
        try:
            download_file(info["url"], dest)
            print(f"  -> Saved to {dest}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {fault_type}: {e}")
            print(f"  You can manually download from: {info['url']}")

    print("=" * 60)
    print("Download complete.")


if __name__ == "__main__":
    download_cwru_dataset()
