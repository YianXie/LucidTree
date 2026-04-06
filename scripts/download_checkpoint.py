import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://drive.google.com/uc?export=download&id=1aEjZLND1T1rgDNQqe_FFmpd_mLAuATYe"
)
CHECKPOINT_PATH = "./models/checkpoint_19x19.pt"

checkpoint_path = Path(CHECKPOINT_PATH)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint_path)

print(f"Downloaded checkpoint to {checkpoint_path}")
