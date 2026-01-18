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
st.caption("Recolor ONLY the background theme while preserving foreground (text/product/buttons) pixel-perfect.")


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Settings")

target_color = st.sidebar.color_picker("Target Theme Color", value="#F3E6D8")
hex_input = st.sidebar.text_input("HEX Input (optional override)", value=target_color).strip()

final_color = hex_input if hex_input else target_color

# HEX Validation (Improvement)
if not (final_color.startswith("#") and len(final_color) == 7):
    st.sidebar.error("Invalid HEX! Use format like #F3E6D8")

strategy = st.sidebar.selectbox(
    "Recolor Strategy",
    ["LAB (Preserve Gradient)", "HSV Hue Shift", "Overlay Blend (Approx)"],
    index=0
)

strength = st.sidebar.slider("Recolor Strength (theme intensity)", 0.0, 1.0, 0.75, 0.01)

st.sidebar.subheader("Mask Settings")
feather_px = st.sidebar.slider("Mask Feather (edge smoothing)", 0, 15, 3, 1)
dilate_px = st.sidebar.slider("Mask Dilation (shadow preservation)", 0, 6, 1, 1)

st.sidebar.subheader("Edge Safety (Halo Fix)")
safe_bg_erode_px = st.sidebar.slider("Background Safe Zone (px)", 0, 6, 1, 1)

st.sidebar.subheader("Contrast Preservation")
preserve_contrast = st.sidebar.checkbox("Preserve Background Contrast (CLAHE)", value=True)
clahe_clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)

st.sidebar.markdown("---")
preview_mode = st.sidebar.radio("Preview Mode", ["Side-by-side", "Toggle (Before/After)"], index=0)

st.sidebar.write("✅ Output format: PNG")
st.sidebar.write("✅ Batch download: ZIP")


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
# Mask Cache (Improvement #6)
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
    progress = st.progress(0)

    for i, f in enumerate(uploaded_files):
        file_bytes = f.getvalue()

        pil_img = Image.open(f).convert("RGB")
        original_bgr = pil_to_bgr(pil_img)

        # 1) Cached Mask base
        alpha = get_cached_mask(file_bytes, pil_img)

        # 2) Refine mask (this depends on sliders, so NOT cached)
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

        # Preview first 5 images only
        if i < 5:
            if preview_mode == "Side-by-side":
                c1, c2 = st.columns(2)
                with c1:
                    st.image(pil_img, caption=f"Original: {f.name}", use_container_width=True)
                with c2:
                    st.image(out_pil, caption=f"Output: {safe_name}", use_container_width=True)

            else:
                # Toggle Mode (Before/After)
                show_after = st.toggle(f"Show AFTER for {f.name}", value=True)
                if show_after:
                    st.image(out_pil, caption=f"AFTER: {safe_name}", use_container_width=True)
                else:
                    st.image(pil_img, caption=f"BEFORE: {f.name}", use_container_width=True)

        progress.progress(int(((i + 1) / len(uploaded_files)) * 100))

    st.success("✅ Batch processing complete!")

    zip_bytes = images_to_zip_bytes(results_for_zip)

    st.download_button(
        label="⬇️ Download Output ZIP",
        data=zip_bytes,
        file_name="converted_banners.zip",
        mime="application/zip"
    )
