"""Functions to convert color images to grayscale images.

Use [coefficients](https://www.mathworks.com/help/matlab/ref/im2gray.html?s_tid=srchtitle_site_search_1_im2gray#mw_0668b600-0226-4640-9ffc-8698ce324f94) from MATLAB.
"""  # noqa: E501

import numpy as np
from numpy.typing import NDArray

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


_COLOR_IMAGE_SHAPE = 3
"""Shape of a NumPy array of a color image."""
_COLOR_IMAGE_CHANNELS = 3
"""Channel number of a NumPy array of a color image."""


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def im2gray(image: NDArray[np.number]) -> NDArray[np.number]:
    """Ensure that the image is in grayscale, convert it otherwise.

    The function used to convert RGB to grayscale is `rgb2gray`.

    Args:
        image: An array of an image.

    Returns:
        The image in grayscale.
    """
    if (
        len(image.shape) == _COLOR_IMAGE_SHAPE
        and image.shape[2] == _COLOR_IMAGE_CHANNELS
    ):
        grayscale_image = rgb2gray(image)
    else:
        grayscale_image = image
    return grayscale_image


def rgb2gray(image: NDArray[np.number]) -> NDArray[np.float64]:
    """Convert an RGB image to grayscale using MATLAB-like coefficients.

    Args:
        image: A 3D numpy array representing the RGB image.

    Returns:
        A 2D numpy array representing the grayscale image.
    """
    coeffs = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    return np.round(np.dot(image[..., :3], coeffs))
