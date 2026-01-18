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


def _target_lab_from_hex(hex_color: str) -> np.ndarray:
    r, g, b = hex_to_rgb(hex_color)
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32)


def _target_hsv_from_hex(hex_color: str) -> np.ndarray:
    r, g, b = hex_to_rgb(hex_color)
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv.astype(np.float32)


def _safe_bg_mask(alpha_fg: np.ndarray, safe_bg_erode_px: int) -> np.ndarray:
    """
    Creates a 'safe background' mask to avoid halo near FG edges.
    This shrinks the background region slightly so we recolor bg pixels
    that are away from the foreground boundary.
    """
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)
    bg = (1.0 - alpha_fg)

    if safe_bg_erode_px <= 0:
        return bg

    k = 2 * safe_bg_erode_px + 1
    kernel = np.ones((k, k), np.uint8)

    bg_u8 = (bg * 255).astype(np.uint8)
    bg_u8 = cv2.erode(bg_u8, kernel, iterations=1)
    bg = bg_u8.astype(np.float32) / 255.0
    return np.clip(bg, 0.0, 1.0)


def _apply_clahe_to_L(L: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Contrast preservation enhancement for the lightness channel (LAB L).
    Works great for gradients looking 'rich' after recolor.
    """
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(L_u8).astype(np.float32)


def recolor_background(
    original_bgr: np.ndarray,
    alpha_fg: np.ndarray,
    target_hex: str,
    strength: float = 0.75,
    strategy: str = "LAB (Preserve Gradient)",
    safe_bg_erode_px: int = 1,
    preserve_contrast: bool = True,
    clahe_clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Recolors background only, keeps FG pixels untouched.

    strategy:
      - "LAB (Preserve Gradient)"  -> best default
      - "HSV Hue Shift"            -> faster but can look weaker
      - "Overlay Blend (Approx)"   -> stylized, strong recolor

    safe_bg_erode_px:
      - halo reduction near foreground edges (recommended 1 or 2)

    preserve_contrast:
      - runs CLAHE on background L channel so gradients retain depth
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    alpha_fg = np.clip(alpha_fg, 0.0, 1.0).astype(np.float32)

    # Base BG mask
    alpha_bg = 1.0 - alpha_fg

    # Safety BG mask (avoid recoloring too close to FG edges)
    bg_safe = _safe_bg_mask(alpha_fg, safe_bg_erode_px=safe_bg_erode_px)

    # convert original to RGB
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    if strategy == "LAB (Preserve Gradient)":
        lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        target_lab = _target_lab_from_hex(target_hex)
        tgt_a = float(target_lab[0, 0, 1])
        tgt_b = float(target_lab[0, 0, 2])

        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # optional contrast preserve for background gradient depth
        if preserve_contrast:
            L2 = _apply_clahe_to_L(L, clip_limit=clahe_clip_limit)
            # Apply only to background; keep FG L intact
            L = L * (1.0 - alpha_bg) + L2 * alpha_bg

        # push bg chroma towards target while retaining lighting (L)
        mix = bg_safe * strength

        A_new = A * (1.0 - mix) + tgt_a * mix
        B_new = B * (1.0 - mix) + tgt_b * mix

        lab_new = np.stack([L, A_new, B_new], axis=2)
        lab_new = np.clip(lab_new, 0, 255).astype(np.uint8)

        recolored_rgb = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
        recolored_bgr = cv2.cvtColor(recolored_rgb, cv2.COLOR_RGB2BGR)

    elif strategy == "HSV Hue Shift":
        hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        target_hsv = _target_hsv_from_hex(target_hex)
        tgt_h = float(target_hsv[0, 0, 0])
        tgt_s = float(target_hsv[0, 0, 1])

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        mix = bg_safe * strength

        H_new = H * (1.0 - mix) + tgt_h * mix
        S_new = S * (1.0 - mix) + tgt_s * mix

        hsv_new = np.stack([H_new, S_new, V], axis=2)
        hsv_new = np.clip(hsv_new, 0, 255).astype(np.uint8)

        recolored_rgb = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
        recolored_bgr = cv2.cvtColor(recolored_rgb, cv2.COLOR_RGB2BGR)

    elif strategy == "Overlay Blend (Approx)":
        # Creates a colored overlay and blends it only on BG
        r, g, b = hex_to_rgb(target_hex)
        overlay_rgb = np.zeros_like(original_rgb, dtype=np.float32)
        overlay_rgb[:, :, 0] = r
        overlay_rgb[:, :, 1] = g
        overlay_rgb[:, :, 2] = b

        base = original_rgb.astype(np.float32)

        # Approx overlay blend formula
        base_norm = base / 255.0
        overlay_norm = overlay_rgb / 255.0

        out = np.where(
            base_norm < 0.5,
            2 * base_norm * overlay_norm,
            1 - 2 * (1 - base_norm) * (1 - overlay_norm)
        )

        out_rgb = (out * 255.0).astype(np.float32)

        mix = (bg_safe * strength)[:, :, None]
        recolored_rgb = base * (1.0 - mix) + out_rgb * mix
        recolored_rgb = np.clip(recolored_rgb, 0, 255).astype(np.uint8)

        recolored_bgr = cv2.cvtColor(recolored_rgb, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Final composite: FG must be original pixels ALWAYS
    a = alpha_fg[:, :, None]
    out_bgr = a * original_bgr.astype(np.float32) + (1.0 - a) * recolored_bgr.astype(np.float32)
    return np.clip(out_bgr, 0, 255).astype(np.uint8)
