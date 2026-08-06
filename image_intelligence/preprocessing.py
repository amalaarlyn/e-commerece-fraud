"""
preprocessing.py

Resizes, normalizes, and standardizes uploaded customer damage images for
downstream feature extraction, histogram comparison, and manipulation analysis.
"""

from typing import Union, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2


def load_image(image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Loads an image input (file path, PIL Image, or numpy array) into a BGR numpy array.
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Unable to read image file at path: {image_input}")
        return img_bgr

    if isinstance(image_input, Image.Image):
        rgb_array = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        if image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        return image_input.copy()

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def resize_and_normalize(
    img_bgr: np.ndarray, target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Resizes a BGR numpy array to the standardized target_size.
    """
    return cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)


def preprocess_image(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    """
    Main entry point for image standardization.

    Returns:
        dict containing:
          - 'bgr': BGR numpy array (512x512)
          - 'rgb': RGB numpy array (512x512)
          - 'gray': Grayscale numpy array (512x512)
          - 'shape': (height, width, channels)
          - 'original_shape': tuple of input image shape before resizing
    """
    raw_bgr = load_image(image_input)
    original_shape = raw_bgr.shape

    resized_bgr = resize_and_normalize(raw_bgr, target_size=target_size)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    resized_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "bgr": resized_bgr,
        "rgb": resized_rgb,
        "gray": resized_gray,
        "shape": resized_bgr.shape,
        "original_shape": original_shape,
    }
