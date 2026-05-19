---
noteId: "91a9f2a0532711f194418f7638caa003"
tags: []

---

# colorization-metrics-ipol

IPOL demo wrapper for the paper **"A survey of metrics used for image colorization assessment"** by Nicolas Maignan, Fabien Pierre, and Frédéric Sur (Université de Lorraine, CNRS, Inria, LORIA).

- **Demo ID:** `77777000567`
- **Demo URL:** <https://ipolcore.ipol.im/cp2/showDemo?demo_id=77777000567>
- **Algorithm source:** <https://gitlab.univ-lorraine.fr/maignan2/colorization-metrics>
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
├── requirements.txt      Pip deps (pulls colorization-metrics + Inria git deps)
└── run.sh                Shell wrapper invoked by ddl.json's `run` field
```

The live DDL is maintained in the IPOL control panel — `DDL.json` here is a frozen reference for reviewers and for `git diff`-able history.

## Pretrained model weights

The reference-based metrics (LPIPS, FID) and the referenceless MANIQA metric require pretrained network weights. To avoid slow runtime downloads inside the IPOL container, weights are delivered as an **IPOL demoExtras** blob (a zip uploaded once via the demo's admin panel). At runtime, `run.sh` exports `TORCH_HOME` and `MANIQA_CHECKPOINT` so that `lpips`, `pytorch_fid`, and `maniqa` find the pre-extracted files under `$demoextras/`.

Expected layout inside the demoExtras zip:

```
torch/
  hub/checkpoints/
    alexnet-owt-*.pth          # LPIPS:alex
    vgg16-*.pth                # LPIPS:vgg
    squeezenet1_1-*.pth        # LPIPS:squeeze
    pt_inception-2015-12-05-*.pth   # pytorch-fid InceptionV3
maniqa/
  ckpt_koniq10k.pt             # MANIQA pretrained checkpoint
```

To regenerate the zip locally, run `scripts/fetch_weights.py` from the parent editor workspace (not committed in this repo).

## How to run outside IPOL

The demo can be exercised locally without IPOL. Assuming `git`, `ffmpeg`, and `python3.11` (or newer) are installed:

```shell
# 1. Get the demo repo (this one).
git clone <demo-repo-url> colorization-metrics-ipol
cd colorization-metrics-ipol

# 2. Install Python deps (uses the algorithm from upstream gitlab).
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) Pre-populate the weights cache so LPIPS/FID/MANIQA work offline.
#    Either run scripts/fetch_weights.py from the editor workspace, or let the
#    libraries download on first invocation:
export TORCH_HOME=~/.cache/torch

# 4. Run on a single image (referenceless metrics only).
bash run.sh "$PWD" path/to/colorized.png /dev/null lab alex 64 3

# 5. Or run paired (all eight metrics).
bash run.sh "$PWD" path/to/colorized.png path/to/ground_truth.png lab alex 64 3
```

## License

MIT (inherited from the upstream algorithm).

## Citation

If you use this demo, please cite the companion IPOL article (see the demo page for the canonical reference).
