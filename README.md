---
noteId: "85808ef0537511f194418f7638caa003"
tags: []

---

# colorization-metrics-ipol

IPOL demo wrapper for the paper **"A survey of metrics used for image colorization assessment"** by Nicolas Maignan, Fabien Pierre, and Frédéric Sur (Université de Lorraine, CNRS, Inria, LORIA).

- **Demo ID:** `77777000567`
- **Demo URL:** <https://ipolcore.ipol.im/cp2/showDemo?demo_id=77777000567>
- **Algorithm source (upstream):** <https://gitlab.univ-lorraine.fr/maignan2/colorization-metrics>
- **Paper preprint:** see the article page linked from the demo.

## What the demo does

Given an uploaded colorized image, the demo evaluates it against a panel of image-quality metrics. Two usage modes are supported:

- **Single-image mode:** upload only the colorized image. The four *referenceless* metrics are computed: **BRISQUE**, **NIQE**, **MANIQA**, **Colorfulness**.
- **Paired mode:** upload the colorized image *and* its ground-truth color reference. The four *reference-based* metrics are additionally computed: **PSNR**, **SSIM**, **LPIPS**, **FID**.

Higher is better for PSNR, SSIM, MANIQA, and Colorfulness; lower is better for LPIPS, FID, BRISQUE, and NIQE.

## Repository layout

```
.
├── .ipol/
│   ├── Dockerfile        IPOL build image (Python 3.11 + PyTorch base)
│   └── packages.txt      Additional apt packages
├── DDL.json              Reference copy of the canonical Demo Description Line
├── README.md             This file
├── main.py               Python entry point used by run.sh
├── requirements.txt      Optional ad-hoc deps (intentionally empty by default)
├── run.sh                Shell wrapper invoked by ddl.json's `run` field
└── src/                  Vendored sources installed into the Docker image
    ├── colorization-metrics/   the algorithm (mirror of gitlab.univ-lorraine.fr/maignan2/colorization-metrics)
    ├── MANIQA/                 vendored from gitlab.inria.fr/nmaignan/MANIQA (one CPU-compat patch applied)
    └── matpy/                  vendored from gitlab.inria.fr/nmaignan/matpy
```

The three vendored directories under `src/` are exact mirrors of the upstream sources at the commits pinned in the algorithm's `pyproject.toml`, with **one** local edit applied to `src/MANIQA/src/maniqa/inference.py` (see "MANIQA CPU patch" below).

The live DDL is maintained in the IPOL control panel — `DDL.json` here is a frozen reference for reviewers and for `git diff`-able history.

## How runtime is kept fast

All pretrained weights are baked into the Docker image during `docker build`, not fetched at request time:

| Metric | Weights location inside the image | Source URL |
| --- | --- | --- |
| LPIPS (alex / vgg / squeeze) | `/home/ipol/.cache/torch/hub/checkpoints/{alexnet,vgg16,squeezenet1_1}-*.pth` | `download.pytorch.org/models/…` via `torchvision.models.*(weights=...)` |
| FID Inception V3 | `/home/ipol/.cache/torch/hub/checkpoints/pt_inception-…pth` | `pytorch_fid.inception.InceptionV3` |
| MANIQA | `/home/ipol/.cache/maniqa/ckpt_koniq10k.pt` | `https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt` (matches MANIQA's `user_cache_dir("maniqa", "nifra")` path) |
| BRISQUE / NIQE | bundled inside `src/colorization-metrics/data/` (committed) | n/a |

So a request inside the IPOL container does no network IO. There is **no** IPOL `demoExtras` dependency.

## MANIQA CPU patch

The MANIQA Koniq10k checkpoint was saved on a CUDA machine. The IPOL runner is CPU-only, so `torch.load(ckpt)` raises *"Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"*. Lines 142–151 of `src/MANIQA/src/maniqa/inference.py` are patched here to (a) load with `map_location=device` and (b) move patches to the same `device` (CPU when CUDA is unavailable). Refresh the vendored copy with `git diff` if you re-pull from upstream.

## How to run outside IPOL

The demo can be exercised locally without IPOL. Requires `git`, `ffmpeg`, and Python 3.11+.

```shell
git clone <demo-repo-url> colorization-metrics-ipol
cd colorization-metrics-ipol

# Create an isolated env and install the vendored sources.
python3 -m venv .venv
source .venv/bin/activate
pip install ./src/matpy ./src/MANIQA ./src/colorization-metrics

# Run on a single image (referenceless metrics only).
bash run.sh "$PWD" path/to/colorized.png /dev/null lab alex 64 3

# Or run paired (all eight metrics).
bash run.sh "$PWD" path/to/colorized.png path/to/ground_truth.png lab alex 64 3
```

The first run will fetch the LPIPS / FID / MANIQA weights into your user cache (~1.4 GB combined). Subsequent runs are offline.

## License

MIT (inherited from the upstream algorithm).

## Citation

If you use this demo, please cite the companion IPOL article (see the demo page for the canonical reference).
