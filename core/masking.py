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

def refine_alpha_mask_with_grabcut(original_bgr, alpha, feather_px=3, dilate_px=1):
    """
    Improve U2Net mask using GrabCut refinement.
    Keeps UI elements (buttons/text blocks) much better.
    """

    h, w = alpha.shape[:2]

    # Convert alpha -> sure FG / sure BG
    # Start by thresholding
    fg_thresh = 0.65
    bg_thresh = 0.10

    grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    grabcut_mask[alpha <= bg_thresh] = cv2.GC_BGD
    grabcut_mask[alpha >= fg_thresh] = cv2.GC_FGD

    # Optional dilation to keep shadows & UI blocks
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)

        sure_fg = (grabcut_mask == cv2.GC_FGD).astype(np.uint8) * 255
        sure_fg = cv2.dilate(sure_fg, kernel, iterations=1)
        grabcut_mask[sure_fg == 255] = cv2.GC_FGD

    # Run GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        original_bgr,
        grabcut_mask,
        None,
        bgdModel,
        fgdModel,
        3,
        cv2.GC_INIT_WITH_MASK
    )

    # Extract final FG mask
    final_fg = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        1.0,
        0.0
    ).astype(np.float32)

    # Feather edges
    if feather_px > 0:
        if feather_px % 2 == 0:
            feather_px += 1
        final_fg = cv2.GaussianBlur(final_fg, (feather_px, feather_px), 0)

    return np.clip(final_fg, 0.0, 1.0)