"""Image resizing module based on interpolation methods like bicubic and bilinear.

This code is adapted from https://github.com/fatheral/matlab_imresize (MIT License).

It provides functions to resize an image using different scaling or output size parameters.
The module includes vectorized implementations and explicit kernel functions for interpolation.
"""  # noqa: E501

from collections.abc import Callable
from enum import Enum
from math import ceil

import numpy as np
from numpy.typing import NDArray

# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class InterpolationMethod(Enum):
    """Enumeration for interpolation methods used in image resizing.

    Attributes:
        BICUBIC: Represents bicubic interpolation.
        BILINEAR: Represents bilinear interpolation.
    """

    BICUBIC = "bicubic"
    BILINEAR = "bilinear"


class ResizeMode(Enum):
    """Enumeration for resizing modes.

    Attributes:
        ORIGINAL: Represents the original resizing mode.
        VECTORIZED: Represents the vectorized resizing mode.
    """

    ORIGINAL = "org"
    VECTORIZED = "vec"


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def imresize(
    image: NDArray[np.float64],
    new_size: float | tuple[int, int],
    interpolation_method: InterpolationMethod = InterpolationMethod.BICUBIC,
    mode: ResizeMode = ResizeMode.VECTORIZED,
) -> NDArray[np.float64]:
    """Resize an image using bicubic or bilinear interpolation.

    Args:
        image: Input image array.
        new_size: Either a scale factor (float) or a target size (height, width).
        interpolation_method: Interpolation method, default is bicubic.
        mode: Resizing mode, either 'vec' (vectorized) or 'org' (original).

    Returns:
        Resized image array.

    Raises:
        ValueError: If an unrecognized interpolation method is provided.
        TypeError: If `new_size` is not a float or tuple of (int, int).
    """
    # Select kernel based on interpolation method
    match interpolation_method:
        case InterpolationMethod.BICUBIC:
            kernel = _cubic_kernel
        case InterpolationMethod.BILINEAR:
            kernel = _triangle_kernel
        case _:
            msg = f"Unrecognized interpolation method: {interpolation_method}"
            raise ValueError(msg)

    # Determine scale and output size
    match new_size:
        case float():
            scale = np.array((new_size, new_size))
            output_size = _compute_size_from_scale(image.shape, scale)
        case (int(), int()):
            scale = np.array(_compute_scale_from_size(image.shape, new_size))
            output_size = new_size
        case _:
            msg = f"Invalid type for new_size: {type(new_size).__name__}.\
                Expected float or tuple[int, int]."
            raise TypeError(msg)

    kernel_width = 4.0
    order = np.argsort(scale)

    # Compute contributions (weights and indices) for both dimensions
    weights, indices = [], []
    for k in range(2):
        w, ind = _compute_contributions(
            image.shape[k], output_size[k], scale[k], kernel, kernel_width
        )
        weights.append(w)
        indices.append(ind)

    # Resize image along each dimension
    resized_image = np.copy(image)
    flag2d = resized_image.ndim == 2  # noqa: PLR2004
    if flag2d:
        resized_image = np.expand_dims(resized_image, axis=2)

    for k in range(2):
        dim = order[k]
        resized_image = _resize_along_dimension(
            resized_image, dim, weights[dim], indices[dim], mode
        )

    if flag2d:
        resized_image = np.squeeze(resized_image, axis=2)

    return resized_image


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _convert_double_to_uint8(image: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Convert an image from double precision (0.0 - 1.0) to uint8 (0 - 255).

    Args:
        image: Image in double precision format.

    Returns:
        Image in uint8 format.
    """
    return np.round(np.clip(image * 255, 0, 255)).astype(np.uint8)


def _compute_size_from_scale(
    old_shape: tuple[int, int], scale: tuple[float, float]
) -> tuple[int, int]:
    """Compute the output image size based on the scale factors and input image shape.

    Args:
        old_shape: Shape of the input image (height, width).
        scale: Scaling factors for height and width.

    Returns:
        Output image size (height, width).
    """
    return (ceil(scale[0] * old_shape[0]), ceil(scale[1] * old_shape[1]))


def _compute_scale_from_size(
    old_shape: tuple[int, int], new_shape: tuple[int, int]
) -> tuple[float, float]:
    """Compute the scaling factors required to resize the image to the given shape.

    Args:
        old_shape: Shape of the input image (height, width).
        new_shape: Desired shape of the output image (height, width).

    Returns:
        Scaling factors for height and width.
    """
    return (new_shape[0] / old_shape[0], new_shape[1] / old_shape[1])


def _triangle_kernel(x: NDArray[np.number]) -> NDArray[np.float64]:
    """Triangle interpolation kernel, used for bilinear interpolation.

    Args:
        x: Input array representing distance from the center pixel.

    Returns:
        Array of interpolation weights.
    """
    y = np.array(x, dtype=np.float64)
    lessthanzero = np.logical_and((y >= -1), y < 0)
    greaterthanzero = np.logical_and((y <= 1), y >= 0)
    return np.multiply((y + 1), lessthanzero) + np.multiply((1 - y), greaterthanzero)


def _cubic_kernel(x: NDArray[np.number]) -> NDArray[np.float64]:
    """Cubic interpolation kernel, used for bicubic interpolation.

    Args:
        x: Input array representing distance from the center pixel.

    Returns:
        Array of interpolation weights.
    """
    y = np.array(x, dtype=np.float64)
    absx = np.abs(y)
    absx2 = absx**2
    absx3 = absx**3
    return np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
        (absx > 1) & (absx <= 2),  # noqa: PLR2004
    )


def _compute_contributions(
    input_length: int,
    output_length: int,
    scale: float,
    kernel: Callable[[NDArray[np.number]], NDArray[np.float64]],
    kernel_width: float,
) -> tuple[NDArray[np.float64], NDArray[int]]:
    """Compute the interpolation weights and corresponding input indices for resizing.

    This function calculates the weights for each pixel in the output image
    based on the interpolation kernel and scales the input image dimensions.

    Args:
        input_length: Size of the input image along the dimension being resized.
        output_length: Size of the output image along the same dimension.
        scale: Scaling factor for resizing.
        kernel: Kernel function for interpolation (e.g., cubic or triangle).
        kernel_width: Width of the interpolation kernel.

    Returns:
        tuple:
            - weights: Weights for each pixel in the output image.
            - indices: Corresponding input indices for each pixel.
    """
    # Adjust kernel based on the scaling factor
    if scale < 1:

        def h(x: NDArray[np.number]) -> NDArray[np.float64]:
            """Apply the kernel function scaled by the scaling factor."""
            return scale * kernel(scale * x)

        kernel_width_used = kernel_width / scale
    else:
        h = kernel
        kernel_width_used = kernel_width

    # Compute the output pixel positions
    x = np.arange(1, output_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)

    # Calculate the left indices for interpolation
    left = np.floor(u - kernel_width_used / 2)
    p = ceil(kernel_width_used) + 2  # Padding for the kernel width
    ind = (
        np.expand_dims(left, axis=1) + np.arange(p) - 1
    )  # Adjust for zero-based indexing
    indices = ind.astype(np.int32)

    # Calculate weights using the kernel function
    weights = h(
        np.expand_dims(u, axis=1) - indices - 1
    )  # Adjust for zero-based indexing
    weights = np.divide(
        weights, np.expand_dims(np.sum(weights, axis=1), axis=1)
    )  # Normalize weights

    # Prepare auxiliary indices for circular indexing
    aux = np.concatenate(
        (np.arange(input_length), np.arange(input_length - 1, -1, step=-1))
    ).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]  # Circular indexing

    # Filter non-zero weights
    ind2store = np.nonzero(np.any(weights, axis=0))
    return weights[:, ind2store], indices[:, ind2store]


def _imresizemex(
    image: NDArray[np.float64] | NDArray[np.uint8],
    weights: NDArray[np.float64],
    indices: NDArray[int],
    dimension: int,
) -> NDArray[np.float64] | NDArray[np.uint8]:
    """Resize image along a specified dimension using matrix multiplication.

    Args:
        image: Input image array.
        weights: Interpolation weights for resizing.
        indices: Corresponding input indices for resizing.
        dimension: Dimension along which to resize (0 for height, 1 for width).

    Returns:
        Resized image along the specified dimension.
        It will be of the same data type as the input image.
    """
    resized_shape = list(image.shape)
    resized_shape[dimension] = weights.shape[0]
    resized_image = np.zeros(resized_shape)

    match dimension:
        case 0:
            for i_img in range(image.shape[1]):
                for i_w in range(weights.shape[0]):
                    w = weights[i_w, :]
                    ind = indices[i_w, :]
                    im_slice = image[ind, i_img].astype(np.float64)
                    resized_image[i_w, i_img] = np.sum(
                        np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0
                    )
        case 1:
            for i_img in range(image.shape[0]):
                for i_w in range(weights.shape[0]):
                    w = weights[i_w, :]
                    ind = indices[i_w, :]
                    im_slice = image[i_img, ind].astype(np.float64)
                    resized_image[i_img, i_w] = np.sum(
                        np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0
                    )
        case _:
            msg = "Dimension must be 0 (height) or 1 (width)."
            raise ValueError(msg)

    if image.dtype == np.uint8:
        resized_image = np.clip(resized_image, 0, 255)
        return np.around(resized_image).astype(np.uint8)

    return resized_image


def _imresizevec(
    image: NDArray[np.float64],
    weights: NDArray[np.float64],
    indices: NDArray[int],
    dimension: int,
) -> NDArray[np.float64]:
    """Vectorized image resizing along a specified dimension.

    This function performs image resizing by applying interpolation weights
    along the specified dimension (height or width) using vectorized operations.

    Args:
        image: Input image array of shape (H, W, C).
        weights: Interpolation weights for resizing of shape (N, C, M).
        indices: Corresponding input indices for resizing.
        dimension: Dimension along which to resize (0 for height, 1 for width).

    Returns:
        Resized image along the specified dimension.
    """
    match dimension:
        case 0:
            weights = weights.reshape((weights.shape[0], weights.shape[2], 1, 1))
            outimg = np.sum(
                weights * image[indices].squeeze(axis=1).astype(np.float64), axis=1
            )
        case 1:
            weights = weights.reshape((1, weights.shape[0], weights.shape[2], 1))
            outimg = np.sum(
                weights * image[:, indices].squeeze(axis=2).astype(np.float64), axis=2
            )
        case _:
            msg = "Dimension must be 0 (height) or 1 (width)."
            raise ValueError(msg)

    # Handle conversion for uint8 images
    if image.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)  # Clip values to valid uint8 range
        return np.round(outimg).astype(np.uint8)

    return outimg


def _resize_along_dimension(
    image: NDArray[np.float64],
    dimension: int,
    weights: NDArray[np.float64],
    indices: NDArray[int],
    mode: ResizeMode = ResizeMode.VECTORIZED,
) -> NDArray[np.float64]:
    """Resize the image along a specified dimension.

    Args:
        image: Input image array.
        dimension: Dimension along which to resize (0 for height, 1 for width).
        weights: Interpolation weights for resizing.
        indices: Corresponding input indices for resizing.
        mode: Resizing mode, either "vec" (vectorized) or "org" (original).

    Returns:
        Resized image along the specified dimension.
    """
    match mode:
        case ResizeMode.ORIGINAL:
            return _imresizemex(image, weights, indices, dimension)
        case ResizeMode.VECTORIZED:
            return _imresizevec(image, weights, indices, dimension)
        case _:
            msg = "Unrecognized mode supplied"
            raise ValueError(msg)
