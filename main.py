"""IPOL demo entry point for the "colorization-metrics" survey (demo id 77777000567).

Invoked by run.sh, which the DDL's `run` field calls. Thin wrapper that just
parses the user parameters and hands off to compute_metrics. All pretrained
weights (LPIPS backbones, FID InceptionV3, MANIQA Koniq10k) are baked into the
Docker image during build, so this script does no caching or downloading.

When the optional ground-truth image is absent, ``compute_metrics`` automatically
skips PSNR / SSIM / LPIPS / FID and runs only the four referenceless metrics.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="colorization-metrics-ipol")
    parser.add_argument("--colored", required=True, help="Path to the colorized image.")
    parser.add_argument("--gt", default=None, help="Optional path to the ground-truth image.")
    parser.add_argument("--color_space", default="lab", choices=["lab", "rgb"])
    parser.add_argument("--LPIPS_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--fid_dims", type=int, default=64, choices=[64, 192, 768, 2048])
    parser.add_argument("--colorfulness_type", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    # Imports happen here (not at module load) so any algorithm-side cwd
    # assumption set up by run.sh is in place first.
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
