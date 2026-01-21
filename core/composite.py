import numpy as np

def transplant_background(
    source_bgr: np.ndarray,
    target_bg_bgr: np.ndarray,
    alpha_fg: np.ndarray,
) -> np.ndarray:
    """
    Background Transplant:

    output = alpha_fg * source_foreground + (1 - alpha_fg) * target_background

    Args:
      source_bgr:      source banner (BGR, uint8)
      target_bg_bgr:   target background (BGR, uint8) SAME SIZE REQUIRED
      alpha_fg:        (H,W) float mask where 1=foreground

    Returns:
      output_bgr: uint8 composited result
    """
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)

    if source_bgr.shape != target_bg_bgr.shape:
        raise ValueError("Target background must be the same size as source image.")

    a = alpha_fg[:, :, None]
    out = a * source_bgr.astype(np.float32) + (1.0 - a) * target_bg_bgr.astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)
