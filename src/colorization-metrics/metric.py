"""Compute colorization metrics on colorized image or video.

This script computes various image quality metrics, including PSNR, SSIM,
    LPIPS, FID, CDC, and COLORFULNESS. The metrics are computed
    on colored video frames, optionally compared with ground truth frames.

Usage:
    For an image:
    $ `python metric.py -c _yourColorizedImgFile_ -gt _yourGroundTruthImgFile_`
    For a video:
    $ `python metric.py -c _yourColorizedVideoFile_ -gt _yourGroundTruthVideoFile_`
        or
    $ `python metric.py -c _yourColorizationDir_ -gt _yourGroundTruthDir_`

Notes:
    Directories must contain **numbered** image files, for example `img_00024.png`
    or `007.jpg`. This is necessary because some metrics use the temporal order
    of images and need to know how to classify image files.
    As far as the index is concerned, there may be a difference
    between your colorization and the ground truth, e.g. the former starts at 0
    and the latter starts at one. The `pair_step` argument is for such cases.
"""

from colorization_metrics.main_cli import main

if __name__ == "__main__":
    main()
