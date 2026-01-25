import torch

from mini_katago import utils

if __name__ == "__main__":
    root = utils.get_project_root()
    checkpoint = torch.load(root / "models/checkpoint.pt", map_location="cpu")
    print(checkpoint.keys())

    print(checkpoint["epoch"])
    print(checkpoint["best_val_loss"])
