#!/usr/bin/env bash
# IPOL run-line wrapper for the colorization-metrics demo (demo id 77777000567).
# Invoked from ddl.json's "run" field. Working directory at entry is IPOL's
# per-execution input folder, where input_0.png (and optionally input_1.png) live.
#
# Args:
#   $1  SRC_ROOT          absolute path to the cloned demo source ($bin in ddl.json)
#   $2  INPUT_COLORED     filename of the colorized image (e.g. input_0.png)
#   $3  INPUT_GT_NAME     filename of the ground-truth image (e.g. input_1.png);
#                         may be absent on disk if the user uploaded only the
#                         colorized input -- the wrapper falls back to running
#                         the four referenceless metrics only.
#   $4  COLOR_SPACE       PSNR color space: "rgb" or "lab"
#   $5  LPIPS_NET         LPIPS backbone: "alex", "vgg", or "squeeze"
#   $6  FID_DIMS          FID Inception feature dim: 64, 192, 768, or 2048
#   $7  COLORFULNESS_TYPE Colorfulness variant: 1, 2, or 3

set -euo pipefail

SRC_ROOT="$1"
INPUT_COLORED="$2"
INPUT_GT_NAME="$3"
COLOR_SPACE="$4"
LPIPS_NET="$5"
FID_DIMS="$6"
COLORFULNESS_TYPE="$7"

# Point ML weight caches at the demoExtras-provided folder when IPOL exports it,
# else fall back to ~/.cache (works at local-dev time outside IPOL).
if [ -n "${demoextras:-}" ] && [ -d "$demoextras" ]; then
    export TORCH_HOME="$demoextras/torch"
    export MANIQA_CHECKPOINT="$demoextras/maniqa/ckpt_koniq10k.pt"
fi

# Resolve input paths to absolute BEFORE we cd into the source root.
INPUT_COLORED_ABS="$(readlink -f "$INPUT_COLORED")"
INPUT_GT_ABS=""
if [ -f "$INPUT_GT_NAME" ]; then
    INPUT_GT_ABS="$(readlink -f "$INPUT_GT_NAME")"
fi

cd "$SRC_ROOT"

if [ -n "$INPUT_GT_ABS" ]; then
    exec python main.py --colored "$INPUT_COLORED_ABS" --gt "$INPUT_GT_ABS" \
        --color_space "$COLOR_SPACE" --LPIPS_net "$LPIPS_NET" \
        --fid_dims "$FID_DIMS" --colorfulness_type "$COLORFULNESS_TYPE"
else
    exec python main.py --colored "$INPUT_COLORED_ABS" \
        --color_space "$COLOR_SPACE" --LPIPS_net "$LPIPS_NET" \
        --fid_dims "$FID_DIMS" --colorfulness_type "$COLORFULNESS_TYPE"
fi
