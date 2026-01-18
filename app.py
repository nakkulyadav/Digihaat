import streamlit as st
import numpy as np
import cv2
from PIL import Image
import hashlib

from core.masking import get_alpha_mask_u2net, refine_alpha_mask_with_grabcut
from core.recolor import recolor_background
from core.zip_export import images_to_zip_bytes


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def file_hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


st.set_page_config(page_title="Bulk Banner BG Transformer", layout="wide")
st.title("🎨 Bulk Banner Background Theme Transformer")
st.caption("Change ONLY the background theme color while keeping text, product, and CTA buttons unchanged.")


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Theme Color")

target_color = st.sidebar.color_picker("Target Theme Color", value="#F3E6D8")
st.sidebar.caption("Pick the new theme color that should replace the banner background across the whole batch.")

hex_input = st.sidebar.text_input("HEX Input (optional override)", value=target_color).strip()
st.sidebar.caption("Paste an exact color code like #F3E6D8 for brand consistency.")

final_color = hex_input if hex_input else target_color

# HEX Validation
if not (final_color.startswith("#") and len(final_color) == 7):
    st.sidebar.error("Invalid HEX! Use format like #RRGGBB (example: #F3E6D8)")

# Locked strategy (no user selection)
strategy = "LAB (Preserve Gradient)"

strength = st.sidebar.slider("Recolor Strength (theme intensity)", 0.0, 1.0, 0.75, 0.01)
st.sidebar.caption("Controls how strongly the background shifts to the new theme color.")

feather_px = st.sidebar.slider("Mask Feather (edge smoothing)", 0, 15, 3, 1)
st.sidebar.caption("Smoothens the edges between foreground and background to avoid jagged cutouts.")

dilate_px = st.sidebar.slider("Mask Dilation (shadow preservation)", 0, 6, 1, 1)
st.sidebar.caption("Expands foreground protection slightly to preserve soft shadows near the product.")


# Locked internal settings (no UI)
safe_bg_erode_px = 0            # ✅ Edge safety OFF internally
preserve_contrast = True         # ✅ Always preserve contrast
clahe_clip_limit = 2.0           # ✅ Fixed contrast strength

st.sidebar.markdown("---")

# -------------------------
# Upload
# -------------------------
uploaded_files = st.file_uploader(
    "Upload banners (PNG / JPG / WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload multiple banners to start.")
    st.stop()

st.write(f"✅ {len(uploaded_files)} image(s) uploaded")


# -------------------------
# Mask Cache
# -------------------------
if "mask_cache" not in st.session_state:
    st.session_state.mask_cache = {}  # {file_hash: alpha_mask_float32}


def get_cached_mask(file_bytes: bytes, pil_img: Image.Image) -> np.ndarray:
    h = file_hash_bytes(file_bytes)

    if h in st.session_state.mask_cache:
        return st.session_state.mask_cache[h]

    alpha = get_alpha_mask_u2net(pil_img)
    st.session_state.mask_cache[h] = alpha
    return alpha


process_btn = st.button("🚀 Process Batch", type="primary")


# -------------------------
# Process
# -------------------------
if process_btn:
    # prevent crash on invalid hex
    if not (final_color.startswith("#") and len(final_color) == 7):
        st.error("❌ Invalid HEX color. Please enter in format #RRGGBB.")
        st.stop()

    results_for_zip = []

    st.subheader("Preview (first 5 images)")
    st.caption("Left = Original | Right = Background Theme Changed")
    progress = st.progress(0)

    for i, f in enumerate(uploaded_files):
        file_bytes = f.getvalue()

        pil_img = Image.open(f).convert("RGB")
        original_bgr = pil_to_bgr(pil_img)

        # 1) Cached Mask base
        alpha = get_cached_mask(file_bytes, pil_img)

        # 2) Refine mask (depends on sliders)
        alpha_refined = refine_alpha_mask_with_grabcut(
            original_bgr=original_bgr,
            alpha=alpha,
            feather_px=feather_px,
            dilate_px=dilate_px
        )

        # 3) Recolor
        out_bgr = recolor_background(
            original_bgr=original_bgr,
            alpha_fg=alpha_refined,
            target_hex=final_color,
            strength=strength,
            strategy=strategy,
            safe_bg_erode_px=safe_bg_erode_px,
            preserve_contrast=preserve_contrast,
            clahe_clip_limit=clahe_clip_limit,
        )

        out_pil = bgr_to_pil(out_bgr)

        # Save for ZIP
        safe_name = f"{f.name.rsplit('.', 1)[0]}_bg.png"
        results_for_zip.append((safe_name, out_pil))

        # Preview first 5 images only (always side-by-side)
        if i < 5:
            c1, c2 = st.columns(2)
            with c1:
                st.image(pil_img, caption=f"Original: {f.name}", use_container_width=True)
            with c2:
                st.image(out_pil, caption=f"Output: {safe_name}", use_container_width=True)

        progress.progress(int(((i + 1) / len(uploaded_files)) * 100))

    st.success("✅ Batch processing complete!")

    zip_bytes = images_to_zip_bytes(results_for_zip)

    st.download_button(
        label="⬇️ Download Output ZIP",
        data=zip_bytes,
        file_name="converted_banners.zip",
        mime="application/zip"
    )
