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
    r, g, b = hex_to_rgb(hex_color)
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab


def get_white_protect_mask(original_bgr: np.ndarray) -> np.ndarray:
    """
    1 = protect whites (do NOT recolor)
    0 = recolor allowed
    """
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # White-ish pixels: high V, low S
    white_mask = (V > 235) & (S < 25)
    return white_mask.astype(np.float32)


def safe_background_mask(alpha_fg: np.ndarray, safe_bg_erode_px: int = 1) -> np.ndarray:
    """
    Creates a safe BG mask to reduce halos near edges.
    """
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)
    bg = (1.0 - alpha_fg)

    if safe_bg_erode_px <= 0:
        return bg

    k = 2 * safe_bg_erode_px + 1
    kernel = np.ones((k, k), np.uint8)

    bg_u8 = (bg * 255).astype(np.uint8)
    bg_u8 = cv2.erode(bg_u8, kernel, iterations=1)

    return (bg_u8.astype(np.float32) / 255.0)


def make_distance_blend_mask(bg_mask: np.ndarray, fade_px: int = 25) -> np.ndarray:
    """
    Creates a smooth blend mask using distance transform.
    Near FG boundary: 0
    Deep BG: 1
    """
    bg_u8 = (np.clip(bg_mask, 0.0, 1.0) * 255).astype(np.uint8)

    # Distance from background pixels to nearest 0 pixel
    dist = cv2.distanceTransform(bg_u8, distanceType=cv2.DIST_L2, maskSize=3)

    # Normalize with fade_px (cap)
    dist = np.clip(dist / float(max(fade_px, 1)), 0.0, 1.0).astype(np.float32)
    return dist


def recolor_background(
    original_bgr: np.ndarray,
    alpha_fg: np.ndarray,
    target_hex: str,
    strength: float = 0.75,
    strategy: str = "LAB (Preserve Gradient)",
    safe_bg_erode_px: int = 1,
    preserve_contrast: bool = False,
    clahe_clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Gradient-friendly background tone shift:

    ✅ Tone shift only (LAB A/B shift)
    ✅ L channel preserved for original lighting/gradient
    ✅ Whites are protected (no tinting)
    ✅ Smooth blending near FG edges using distance transform
    """
    if strategy != "LAB (Preserve Gradient)":
        raise ValueError("Only 'LAB (Preserve Gradient)' is supported.")

    strength = float(np.clip(strength, 0.0, 1.0))
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)

    alpha_bg = 1.0 - alpha_fg

    # Safe background region (halo prevention)
    bg_safe = safe_background_mask(alpha_fg, safe_bg_erode_px=safe_bg_erode_px)

    # White protection (don’t tint whites/highlights)
    white_mask = get_white_protect_mask(original_bgr)

    # Base recolor allowed area
    recolor_allowed = bg_safe * (1.0 - white_mask)

    # ✅ Gradient blending trick:
    # Fade recolor near FG boundary so it blends naturally
    edge_fade = make_distance_blend_mask(recolor_allowed, fade_px=30)

    # Slight blur on recolor mask to remove hard boundaries
    recolor_allowed_blur = cv2.GaussianBlur(recolor_allowed, (0, 0), sigmaX=2.0)

    # Final mix map
    # - non-linear strength to avoid flattening
    # - edge fade keeps blending natural
    mix = recolor_allowed_blur * edge_fade * (strength ** 1.35)
    mix = np.clip(mix, 0.0, 1.0)

    # Convert image to LAB
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    target_lab = make_target_lab_from_hex(target_hex)
    tgt_a = float(target_lab[0, 0, 1])
    tgt_b = float(target_lab[0, 0, 2])

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Optional contrast enhancement (OFF by default for smoother gradients)
    if preserve_contrast:
        L_u8 = np.clip(L, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
        L_enh = clahe.apply(L_u8).astype(np.float32)
        L = L * (1.0 - alpha_bg) + L_enh * alpha_bg

    # Apply tone shift on background only
    A_new = A * (1.0 - mix) + tgt_a * mix
    B_new = B * (1.0 - mix) + tgt_b * mix

    lab_new = np.stack([L, A_new, B_new], axis=2)
    lab_new = np.clip(lab_new, 0, 255).astype(np.uint8)

    recolored_rgb = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    recolored_bgr = cv2.cvtColor(recolored_rgb, cv2.COLOR_RGB2BGR)

    # Final composite (foreground stays original pixel-perfect)
    a = alpha_fg[:, :, None]
    output_bgr = a * original_bgr.astype(np.float32) + (1.0 - a) * recolored_bgr.astype(np.float32)

    return np.clip(output_bgr, 0, 255).astype(np.uint8)
