import argparse
import os
import random

import numpy as np
import requests
import torch
from platformdirs import user_cache_dir
from skimage.io import imread
from skimage.util import img_as_float32
from torchvision import transforms
from tqdm import tqdm

from maniqa.models.maniqa_net import MANIQA
from maniqa.utils.inference_process import Normalize, ToTensor

APP_NAME = "maniqa"
APP_AUTHOR = "nifra"

WEIGHTS_DIR = user_cache_dir(APP_NAME, APP_AUTHOR)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

NETWORK_WEIGHTS = os.path.join(WEIGHTS_DIR, "ckpt_koniq10k.pt")


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super().__init__()
        self.img_name = image_path.split("/")[-1]

        self.img = np.transpose(img_as_float32(imread(image_path)), (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        new_h = 224
        new_w = 224

        if h < new_h or w < new_w:
            msg = f"Image too small for {new_h}x{new_w} patch extraction."
            raise ValueError(msg)

        self.img_patches = []
        for _i in range(num_crops):
            top = np.random.randint(0, h - new_h)  # noqa: NPY002
            left = np.random.randint(0, w - new_w)  # noqa: NPY002
            patch = self.img[:, top : top + new_h, left : left + new_w]
            self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {"d_img_org": patch, "score": 0, "d_name": self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


def setup_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_score(
    image_path: str, random_seed: int = 20, cpu_nb: int = 1, progress_bar: bool = False
) -> float:
    os.environ["OMP_NUM_THREADS"] = str(cpu_nb)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_nb)
    os.environ["MKL_NUM_THREADS"] = str(cpu_nb)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_nb)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_nb)
    torch.set_num_threads(cpu_nb)

    setup_seed(random_seed)

    config = {
        "image_path": image_path,
        # valid times
        "num_crops": 20,
        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
        # checkpoint path
        "ckpt_path": NETWORK_WEIGHTS,
    }

    if not os.path.exists(config["ckpt_path"]):
        url = "https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt"
        response = requests.get(url, stream=True, timeout=100)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="MANIQA weights download",
            ncols=90,
        )

        with open(config["ckpt_path"], "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                f.write(data)

    # data load
    img = Image(
        image_path=config["image_path"],
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        num_crops=config["num_crops"],
    )

    # model defination
    net = MANIQA(
        embed_dim=config["embed_dim"],
        num_outputs=config["num_outputs"],
        dim_mlp=config["dim_mlp"],
        patch_size=config["patch_size"],
        img_size=config["img_size"],
        window_size=config["window_size"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        num_tab=config["num_tab"],
        scale=config["scale"],
    )

    # IPOL-vendor patch: ckpt was saved on CUDA, container is CPU-only.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(config["ckpt_path"], map_location=device), strict=False)
    net = net.to(device)

    avg_score = 0
    for i in tqdm(range(config["num_crops"]), disable=not progress_bar):
        with torch.no_grad():
            net.eval()
            patch_sample = img.get_patch(i)
            patch = patch_sample["d_img_org"].to(device)
            patch = patch.unsqueeze(0)
            score = net(patch)
            avg_score += score

    avg_score = float(avg_score)
    return avg_score / config["num_crops"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the MANIQA score for an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "-s",
        "--seed",
        default=20,
        type=int,
        help="Seed for the random generator (default : %(default)s).",
    )
    parser.add_argument(
        "-c",
        "--cpu_nb",
        default=1,
        type=int,
        help="Number of CPU cores to use (default : %(default)s).",
    )
    parser.add_argument(
        "-b",
        "--progress_bar",
        default=True,
        type=bool,
        help="Whether to display a progress bar (default : %(default)s).",
    )

    args = parser.parse_args()

    print(
        f"Image score: {infer_score(args.image_path, args.seed, args.cpu_nb, args.progress_bar)}"
    )
