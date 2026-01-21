import streamlit as st
import numpy as np
import cv2
from PIL import Image
import hashlib

from core.masking import (
    get_alpha_mask_u2net,
    refine_alpha_mask_with_grabcut,
    get_ocr_text_mask,
    get_cta_saturation_mask,
    merge_foreground_masks,
    tighten_foreground_mask,
)
from core.recolor import recolor_background
from core.zip_export import images_to_zip_bytes


# -------------------------
# Helpers
# -------------------------
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def file_hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Bulk Banner BG Transformer", layout="wide")
st.title("🎨 Bulk Banner Background Theme Transformer")
st.caption(
    "Batch recolor ONLY the background theme while keeping product, text, and CTA buttons unchanged."
)

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Theme Color")

target_color = st.sidebar.color_picker("Target Theme Color", value="#F3E6D8")
st.sidebar.caption("Select the new background theme color for this batch.")

hex_input = st.sidebar.text_input("HEX Input (optional override)", value=target_color).strip()
st.sidebar.caption("Paste an exact brand HEX like #F3E6D8 for consistent output.")

final_color = hex_input if hex_input else target_color

if not (final_color.startswith("#") and len(final_color) == 7):
    st.sidebar.error("Invalid HEX! Use format like #RRGGBB (example: #F3E6D8)")

strength = st.sidebar.slider("Recolor Strength (theme intensity)", 0.0, 1.0, 0.30, 0.01)
st.sidebar.caption(
    "Controls how strongly the background tone shifts to the target color while keeping gradients natural."
)

st.sidebar.markdown("---")
st.sidebar.header("Foreground Protection")

feather_px = st.sidebar.slider("Mask Feather (edge smoothing)", 0, 15, 1, 1)
st.sidebar.caption("Smoothens mask edges to avoid jagged cutout boundaries.")

dilate_px = st.sidebar.slider("Mask Dilation (shadow preservation)", 0, 6, 0, 1)
st.sidebar.caption("Expands protected foreground slightly to preserve soft shadows around product edges.")

st.sidebar.markdown("---")
st.sidebar.info(
    "✅ OCR Text Protection: ALWAYS ON (protects only text strokes)\n"
    "✅ CTA Protection: ALWAYS ON (strong-color CTA buttons)\n"
    "✅ White Background Protection: ON (prevents tinting whites)\n"
    "✅ Strategy: LAB Tone Shift (preserves gradient lighting)\n"
    "✅ Output: PNG + ZIP"
)

# Locked internal settings for quality
strategy = "LAB (Preserve Gradient)"
safe_bg_erode_px = 1       # halo protection
preserve_contrast = False  # keep gradients smooth (avoid CLAHE flattening)
clahe_clip_limit = 2.0


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
# Caches
# -------------------------
if "mask_cache" not in st.session_state:
    st.session_state.mask_cache = {}  # U2Net alpha cache

if "ocr_cache" not in st.session_state:
    st.session_state.ocr_cache = {}   # OCR mask cache


def get_cached_u2net_mask(file_bytes: bytes, pil_img: Image.Image) -> np.ndarray:
    h = file_hash_bytes(file_bytes)
    if h in st.session_state.mask_cache:
        return st.session_state.mask_cache[h]

    alpha = get_alpha_mask_u2net(pil_img)
    st.session_state.mask_cache[h] = alpha
    return alpha


def get_cached_ocr_mask(file_bytes: bytes, original_bgr: np.ndarray) -> np.ndarray:
    h = file_hash_bytes(file_bytes)
    if h in st.session_state.ocr_cache:
        return st.session_state.ocr_cache[h]

    # IMPORTANT: This OCR mask is stroke-level, not full boxes
    ocr_mask = get_ocr_text_mask(original_bgr, expand_px=1)
    st.session_state.ocr_cache[h] = ocr_mask
    return ocr_mask


# -------------------------
# Process Batch
# -------------------------
process_btn = st.button("🚀 Process Batch", type="primary")

if process_btn:
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

        # 1) Base U2Net mask (cached)
        alpha_u2net = get_cached_u2net_mask(file_bytes, pil_img)

        # 2) GrabCut refinement (depends on feather/dilate)
        alpha_grabcut = refine_alpha_mask_with_grabcut(
            original_bgr=original_bgr,
            alpha=alpha_u2net,
            feather_px=feather_px,
            dilate_px=dilate_px,
        )

        # 3) OCR text stroke mask (cached, always ON)
        ocr_mask = get_cached_ocr_mask(file_bytes, original_bgr)

        # 4) CTA mask (always ON)
        cta_mask = get_cta_saturation_mask(original_bgr)

        # 5) Merge masks (union)
        alpha_final = merge_foreground_masks(alpha_grabcut, ocr_mask, cta_mask)

        # 6) Tighten final mask to reduce any rigid block protection
        alpha_final = tighten_foreground_mask(alpha_final, erode_px=0)

        # 7) Recolor background only (tone shift)
        out_bgr = recolor_background(
            original_bgr=original_bgr,
            alpha_fg=alpha_final,
            target_hex=final_color,
            strength=strength,
            strategy=strategy,
            safe_bg_erode_px=safe_bg_erode_px,
            preserve_contrast=preserve_contrast,
            clahe_clip_limit=clahe_clip_limit,
        )

        out_pil = bgr_to_pil(out_bgr)

        safe_name = f"{f.name.rsplit('.', 1)[0]}_bg.png"
        results_for_zip.append((safe_name, out_pil))

        # preview first 5
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
