#!/usr/bin/env bash
# IPOL run-line wrapper for the colorization-metrics demo (demo id 77777000567).
# Invoked from DDL.json's "run" field. Working directory at entry is IPOL's
# per-execution input folder, where input_0.png (and optionally input_1.png) live.
#
# Pretrained weights live in IPOL demoExtras (uploaded once via the admin panel)
# and are wired up below: TORCH_HOME points at $demoextras/torch for torchvision /
# pytorch-fid, and a symlink bridges MANIQA's hardcoded ~/.cache/maniqa path to
# $demoextras/maniqa.
#
# Args:
#   $1  BIN               absolute path to the cloned demo source ($bin in DDL.json)
#   $2  DEMOEXTRAS        absolute path to demoExtras ($demoextras in DDL.json)
#   $3  INPUT_COLORED     filename of the colorized image (e.g. input_0.png)
#   $4  INPUT_GT_NAME     filename of the ground-truth image (e.g. input_1.png);
#                         may be absent on disk -- we skip -gt in that case so
#                         compute_metrics runs only the referenceless metrics.
#   $5  COLOR_SPACE       PSNR color space: "rgb" or "lab"
#   $6  LPIPS_NET         LPIPS backbone: "alex", "vgg", or "squeeze"
#   $7  FID_DIMS          FID Inception feature dim: 64, 192, 768, or 2048
#   $8  COLORFULNESS_TYPE Colorfulness variant: 1, 2, or 3

set -euo pipefail

# Permet de lister toutes les variables d'environnement actives sur IPOL
env | grep -i extra >&2 || echo "Aucune variable contenant le mot extra n'a été trouvée" >&2

if [ -d "/workdir/demoextras" ]; then
    echo "Le dossier /workdir/demoextras existe bien." >&2
    echo "Voici la liste des fichiers et dossiers trouvés à l'intérieur :" >&2
    ls -la /workdir/demoextras >&2
else
    echo "ERREUR : Le dossier /workdir/demoextras n'existe pas du tout sur le disque !" >&2
fi

BIN="$1"
DEMOEXTRAS="$2"
INPUT_COLORED="$3"

## IPOL décale les arguments si input_1.png est absent.
## On détecte si le 4ème argument est un fichier OU si le 5ème est l'espace de couleur.
#if [ "$5" = "lab" ] || [ "$5" = "rgb" ]; then
#    # Mode Paired : input_1.png est bien présent en 4ème position
#    INPUT_GT_NAME="$4"
#    COLOR_SPACE="$5"
#    LPIPS_NET="$6"
#    FID_DIMS="$7"
#    COLORFULNESS_TYPE="$8"
#else
#    # Mode Single-image : input_1.png a été sauté par IPOL.
#    # Tout le reste a glissé d'un cran vers la gauche.
#    INPUT_GT_NAME="/dev/null"
#    COLOR_SPACE="$4"
#    LPIPS_NET="$5"
#    FID_DIMS="$6"
#    COLORFULNESS_TYPE="$7"
#fi

INPUT_GT_NAME="$4"
COLOR_SPACE="$5"
LPIPS_NET="$6"
FID_DIMS="$7"

echo "\$1 (BIN)               = '$BIN'" >&2
echo "\$2 (DEMOEXTRAS)        = '$DEMOEXTRAS'" >&2
echo "\$3 (INPUT_COLORED)     = '$INPUT_COLORED'" >&2
echo "\$4 (INPUT_GT_NAME)     = '$INPUT_GT_NAME'" >&2
echo "\$5 (COLOR_SPACE)       = '$COLOR_SPACE'" >&2
echo "\$6 (LPIPS_NET)         = '$LPIPS_NET'" >&2
echo "\$7 (FID_DIMS)          = '$FID_DIMS'" >&2

COLORFULNESS_TYPE="$8"

# Defensive guard: fail loudly if demoExtras is missing or stripped, instead of
# letting MANIQA silently start a 543 MB download at request time.
if [ ! -d "$DEMOEXTRAS" ] || [ ! -f "$DEMOEXTRAS/maniqa/ckpt_koniq10k.pt" ]; then
    echo "ERROR: demoExtras is missing or incomplete at '$DEMOEXTRAS'." >&2
    echo "       Upload demoextras.zip via the IPOL admin panel for demo 77777000567." >&2
    exit 1
fi

# Point torchvision / pytorch-fid at the demoExtras-mounted weights cache.
# Layout inside $DEMOEXTRAS: torch/hub/checkpoints/*.pth and maniqa/ckpt_koniq10k.pt
export TORCH_HOME="$DEMOEXTRAS/torch"

# MANIQA's inference.py hardcodes WEIGHTS_DIR = platformdirs.user_cache_dir("maniqa","nifra")
# == /home/ipol/.cache/maniqa with no env-var hook. Symlink it into demoExtras.
mkdir -p /home/ipol/.cache
ln -sfn "$DEMOEXTRAS/maniqa" /home/ipol/.cache/maniqa

# Resolve input paths BEFORE we cd, since IPOL invokes us with cwd = input folder.
INPUT_COLORED_ABS="$(readlink -f "$INPUT_COLORED")"
INPUT_GT_ABS=""
if [ -f "$INPUT_GT_NAME" ]; then
    INPUT_GT_ABS="$(readlink -f "$INPUT_GT_NAME")"
fi

# BRISQUE/NIQE read their bundled models via "data/allmodel" etc. (cwd-relative
# in the algorithm code), so we cd into the vendored algorithm source root.
cd "$BIN/src/colorization-metrics"

if [ -n "$INPUT_GT_ABS" ]; then
    exec python "$BIN/main.py" --colored "$INPUT_COLORED_ABS" --gt "$INPUT_GT_ABS" \
        --color_space "$COLOR_SPACE" --LPIPS_net "$LPIPS_NET" \
        --fid_dims "$FID_DIMS" --colorfulness_type "$COLORFULNESS_TYPE"
else
    exec python "$BIN/main.py" --colored "$INPUT_COLORED_ABS" \
        --color_space "$COLOR_SPACE" --LPIPS_net "$LPIPS_NET" \
        --fid_dims "$FID_DIMS" --colorfulness_type "$COLORFULNESS_TYPE"
fi
