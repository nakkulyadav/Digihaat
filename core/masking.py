import numpy as np
import cv2
from PIL import Image
from rembg import remove
from rapidocr_onnxruntime import RapidOCR


# ✅ OCR engine loaded once
_OCR_ENGINE = RapidOCR()


# -------------------------------------------------------
# 1) Base U2Net Mask
# -------------------------------------------------------
def get_alpha_mask_u2net(pil_img: Image.Image) -> np.ndarray:
    """
    Uses U2Net (via rembg) to produce an alpha matte.
    Returns float32 mask (H,W) in [0,1]
      1 = foreground
      0 = background
    """
    rgba = remove(pil_img.convert("RGB"))
    rgba_np = np.array(rgba)
    alpha = rgba_np[:, :, 3].astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


# -------------------------------------------------------
# 2) GrabCut refinement
# -------------------------------------------------------
def refine_alpha_mask_with_grabcut(
    original_bgr: np.ndarray,
    alpha: np.ndarray,
    feather_px: int = 3,
    dilate_px: int = 1,
    fg_thresh: float = 0.65,
    bg_thresh: float = 0.10,
) -> np.ndarray:
    """
    Refines U2Net alpha mask using GrabCut.
    """
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    h, w = alpha.shape[:2]

    grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[alpha <= bg_thresh] = cv2.GC_BGD
    grabcut_mask[alpha >= fg_thresh] = cv2.GC_FGD

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)

        sure_fg = (grabcut_mask == cv2.GC_FGD).astype(np.uint8) * 255
        sure_fg = cv2.dilate(sure_fg, kernel, iterations=1)
        grabcut_mask[sure_fg == 255] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img=original_bgr,
        mask=grabcut_mask,
        rect=None,
        bgdModel=bgdModel,
        fgdModel=fgdModel,
        iterCount=3,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    final_fg = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        1.0,
        0.0
    ).astype(np.float32)

    if feather_px and feather_px > 0:
        if feather_px % 2 == 0:
            feather_px += 1
        final_fg = cv2.GaussianBlur(final_fg, (feather_px, feather_px), 0)

    return np.clip(final_fg, 0.0, 1.0)


# -------------------------------------------------------
# 3) OCR Text Mask (STROKE LEVEL, NOT FULL BOX)
# Best for dark text on light background
# -------------------------------------------------------
def get_ocr_text_mask(original_bgr: np.ndarray, expand_px: int = 1) -> np.ndarray:
    """
    OCR-based text protection (stroke-level).

    ✅ Uses OCR only to localize text area
    ✅ Extracts ONLY actual text strokes (no rigid rectangle patches)
    ✅ Optimized for dark text on light background
    """
    h, w = original_bgr.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    result, _ = _OCR_ENGINE(rgb)

    if not result:
        return final_mask.astype(np.float32)

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # Blackhat highlights dark text over light background
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)

    # OTSU threshold isolates text strokes
    _, text_candidate = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for item in result:
        box = np.array(item[0], dtype=np.int32)

        # mask only inside OCR polygon
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [box], 255)

        inside = cv2.bitwise_and(text_candidate, text_candidate, mask=poly_mask)
        final_mask = cv2.bitwise_or(final_mask, inside)

    if expand_px > 0:
        k = 2 * expand_px + 1
        kernel = np.ones((k, k), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    return (final_mask > 0).astype(np.float32)


# -------------------------------------------------------
# 4) CTA Strong-color protection
# (stricter threshold to avoid capturing gradient background)
# -------------------------------------------------------
def get_cta_saturation_mask(original_bgr: np.ndarray) -> np.ndarray:
    """
    Protects CTA buttons (strong saturated colors).
    Returns float mask (0/1).
    """
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Stricter thresholds to avoid eating background gradients
    cta_mask = (S > 140) & (V > 60)

    return cta_mask.astype(np.float32)


# -------------------------------------------------------
# 5) Merge utility
# -------------------------------------------------------
def merge_foreground_masks(*masks: np.ndarray) -> np.ndarray:
    """
    Union of all foreground masks (max operation).
    """
    if len(masks) == 0:
        raise ValueError("merge_foreground_masks() needs at least 1 mask.")

    out = np.zeros_like(masks[0], dtype=np.float32)
    for m in masks:
        out = np.maximum(out, m.astype(np.float32))

    return np.clip(out, 0.0, 1.0)


def tighten_foreground_mask(alpha_fg: np.ndarray, erode_px: int = 1) -> np.ndarray:
    """
    Slightly shrinks final foreground protection to avoid rigid blocks.
    Helps keep gradients continuous behind text.
    """
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)

    if erode_px <= 0:
        return alpha_fg

    k = 2 * erode_px + 1
    kernel = np.ones((k, k), np.uint8)

    m = (alpha_fg * 255).astype(np.uint8)
    m = cv2.erode(m, kernel, iterations=1)

    return (m.astype(np.float32) / 255.0)
