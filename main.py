"""IPOL demo entry point for the "colorization-metrics" survey (demo id 77777000567).

Invoked by run.sh (which the DDL `run` field calls). Thin wrapper that:
  - resolves where the pretrained-model weights live (demoExtras when running in
    IPOL, fallback cache when running locally for development),
  - hands off to ``colorization_metrics.evaluate.compute_metrics`` with the user
    parameters.

The algorithm itself decides which metrics apply: when ``gt`` is ``None`` it
computes only the four referenceless metrics (BRISQUE, NIQE, MANIQA,
Colorfulness); when ``gt`` is provided it additionally computes PSNR, SSIM,
LPIPS, and FID.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(prog="colorization-metrics-ipol")
    parser.add_argument("--colored", required=True, help="Path to the colorized image.")
    parser.add_argument("--gt", default=None, help="Optional path to the ground-truth image.")
    parser.add_argument("--color_space", default="lab", choices=["lab", "rgb"])
    parser.add_argument("--LPIPS_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--fid_dims", type=int, default=64, choices=[64, 192, 768, 2048])
    parser.add_argument("--colorfulness_type", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    # Point weight caches at $demoextras when IPOL provides it, otherwise fall
    # back to a sibling demoextras/ folder (useful for local dev).
    demoextras = os.environ.get("demoextras") or os.path.join(ROOT, "demoextras")
    if os.path.isdir(demoextras):
        os.environ.setdefault("TORCH_HOME", os.path.join(demoextras, "torch"))
        maniqa_ckpt = os.path.join(demoextras, "maniqa", "ckpt_koniq10k.pt")
        if os.path.isfile(maniqa_ckpt):
            os.environ.setdefault("MANIQA_CHECKPOINT", maniqa_ckpt)

    # Imports happen after env-var setup so the libraries pick up the cache paths.
    from colorization_metrics.evaluate import MetricParameters, compute_metrics

    compute_metrics(
        colored=args.colored,
        gt=args.gt,
        params=MetricParameters(
            color_space=args.color_space,
            lpips_net=args.LPIPS_net,
            fid_dims=args.fid_dims,
            colorfulness_type=args.colorfulness_type,
        ),
        save_results=False,
    )


if __name__ == "__main__":
    sys.exit(main())
