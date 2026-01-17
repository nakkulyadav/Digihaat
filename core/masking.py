import numpy as np
import cv2
from PIL import Image
from rembg import remove


def get_alpha_mask_u2net(pil_img: Image.Image) -> np.ndarray:
    """
    Uses U2Net (via rembg) to produce an alpha matte.
    Returns:
      alpha: float32 mask of shape (H, W) in range [0, 1]
    """
    rgba = remove(pil_img.convert("RGB"))  # returns PIL Image RGBA
    rgba_np = np.array(rgba)
    alpha = rgba_np[:, :, 3].astype(np.float32) / 255.0
    return alpha


def refine_alpha_mask(
    alpha: np.ndarray,
    feather_px: int = 3,
    dilate_px: int = 1,
) -> np.ndarray:
    """
    Refines alpha to preserve edges + keep soft shadows.
    - feather_px: blur kernel size (odd). 0 disables feathering.
    - dilate_px: dilates mask slightly to include shadows/near edges.
    """
    alpha = np.clip(alpha, 0.0, 1.0)

    # dilation (helps keep subtle shadows as foreground)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)
        alpha_u8 = (alpha * 255).astype(np.uint8)
        alpha_u8 = cv2.dilate(alpha_u8, kernel, iterations=1)
        alpha = alpha_u8.astype(np.float32) / 255.0

    # feather edges (anti-alias)
    if feather_px and feather_px > 0:
        if feather_px % 2 == 0:
            feather_px += 1
        alpha = cv2.GaussianBlur(alpha, (feather_px, feather_px), 0)

    return np.clip(alpha, 0.0, 1.0)
