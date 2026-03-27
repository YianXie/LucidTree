# https://drive.google.com/uc?export=download&id=1aEjZLND1T1rgDNQqe_FFmpd_mLAuATYe

import os
import urllib.request
from pathlib import Path

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL")
CHECKPOINT_PATH = Path("/var/data/checkpoint_19x19.pt")

# Only download if not already exists
if not CHECKPOINT_PATH.exists():
    print("Downloading checkpoint...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)  # type: ignore
    print("Download complete.")
else:
    print("Checkpoint already exists, skipping download.")
