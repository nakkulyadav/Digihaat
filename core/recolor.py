import numpy as np
import cv2


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("HEX color must be in format #RRGGBB")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def make_target_lab_from_hex(hex_color: str) -> np.ndarray:
    """
    Returns LAB value (1x1x3) uint8 for a target color.
    """
    r, g, b = hex_to_rgb(hex_color)
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab


def recolor_background_preserve_gradient(
    original_bgr: np.ndarray,
    alpha_fg: np.ndarray,
    target_hex: str,
    strength: float = 0.85,
) -> np.ndarray:
    """
    Recolors ONLY background while preserving gradient lighting.

    Idea:
    - Convert image to LAB
    - Keep L channel for background pixels (preserves shading/gradient)
    - Move A/B channels toward the target color's A/B (controls theme color)
    - Composite with foreground untouched (via alpha)

    Args:
      original_bgr: (H,W,3) uint8
      alpha_fg:     (H,W) float32 in [0,1] where 1=foreground
      target_hex:   hex color string e.g. "#F3E6D8"
      strength:     0..1 how strongly to push bg toward target color
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    h, w = original_bgr.shape[:2]

    # Background mask
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)
    alpha_bg = 1.0 - alpha_fg

    # Convert original to LAB
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Target LAB
    target_lab = make_target_lab_from_hex(target_hex).astype(np.float32)
    tgt_a = float(target_lab[0, 0, 1])
    tgt_b = float(target_lab[0, 0, 2])

    # Split channels
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Apply recolor only on background pixels
    # A,B shift toward target chroma while keeping L (lighting)
    A_new = A * (1.0 - alpha_bg * strength) + (tgt_a) * (alpha_bg * strength)
    B_new = B * (1.0 - alpha_bg * strength) + (tgt_b) * (alpha_bg * strength)

    lab_new = np.stack([L, A_new, B_new], axis=2).astype(np.uint8)

    recolored_rgb = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    recolored_bgr = cv2.cvtColor(recolored_rgb, cv2.COLOR_RGB2BGR)

    # Final composite: foreground pixels strictly original
    a = alpha_fg[:, :, None]  # (H,W,1)
    out = (
        a * original_bgr.astype(np.float32)
        + (1.0 - a) * recolored_bgr.astype(np.float32)
    )

    return np.clip(out, 0, 255).astype(np.uint8)
