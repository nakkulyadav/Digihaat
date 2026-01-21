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
)
from core.composite import transplant_background
from core.zip_export import images_to_zip_bytes


# -------------------------
# Helpers
# -------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def file_hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="BG Transplant Tool", layout="wide")
st.title("🧩 Bulk Banner Background Transplant Tool")
st.caption(
    "Extract foreground (product/text/CTA) from source banners and paste it onto a target background. "
    "This tool does NOT recolor. It replaces the background completely."
)

st.sidebar.header("Foreground Protection Settings")

feather_px = st.sidebar.slider("Mask Feather (edge smoothing)", 0, 15, 3, 1)
st.sidebar.caption("Smoothens mask edges to avoid jagged cutout boundaries.")

dilate_px = st.sidebar.slider("Mask Dilation (shadow preservation)", 0, 6, 1, 1)
st.sidebar.caption("Expands protected region slightly to preserve soft shadows around the product.")

st.sidebar.markdown("---")
st.sidebar.info(
    "✅ OCR Text Protection: ALWAYS ON\n"
    "✅ CTA Protection: ALWAYS ON\n"
    "⚠️ Target background must be SAME SIZE as source banners\n"
    "✅ Output: PNG + ZIP"
)


# -------------------------
# Uploads
# -------------------------
st.subheader("1) Upload Source Banners (batch)")
source_files = st.file_uploader(
    "Upload source banners (PNG / JPG / WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

st.subheader("2) Upload Target Background (single image)")
target_file = st.file_uploader(
    "Upload target background (PNG / JPG / WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False
)

if not source_files or not target_file:
    st.info("Upload BOTH source banners and a target background to continue.")
    st.stop()

st.write(f"✅ Source banners uploaded: {len(source_files)}")
st.write("✅ Target background uploaded")


# -------------------------
# Read target background
# -------------------------
target_pil = Image.open(target_file).convert("RGB")
target_bgr = pil_to_bgr(target_pil)
target_h, target_w = target_bgr.shape[:2]


# -------------------------
# Caches
# -------------------------
if "u2net_cache" not in st.session_state:
    st.session_state.u2net_cache = {}  # {filehash: alpha_u2net}

if "ocr_cache" not in st.session_state:
    st.session_state.ocr_cache = {}  # {filehash: ocr_mask}


def get_cached_u2net(file_bytes: bytes, pil_img: Image.Image) -> np.ndarray:
    h = file_hash_bytes(file_bytes)
    if h in st.session_state.u2net_cache:
        return st.session_state.u2net_cache[h]
    alpha = get_alpha_mask_u2net(pil_img)
    st.session_state.u2net_cache[h] = alpha
    return alpha


def get_cached_ocr(file_bytes: bytes, source_bgr: np.ndarray) -> np.ndarray:
    h = file_hash_bytes(file_bytes)
    if h in st.session_state.ocr_cache:
        return st.session_state.ocr_cache[h]
    ocr_mask = get_ocr_text_mask(source_bgr, expand_px=1)
    st.session_state.ocr_cache[h] = ocr_mask
    return ocr_mask


# -------------------------
# Process Button
# -------------------------
process_btn = st.button("🚀 Transplant Background (Batch)", type="primary")

if process_btn:
    results_for_zip = []

    st.subheader("Preview (first 5 images)")
    st.caption("Left = Source | Right = Output (foreground pasted on target background)")
    progress = st.progress(0)

    for i, f in enumerate(source_files):
        file_bytes = f.getvalue()

        src_pil = Image.open(f).convert("RGB")
        src_bgr = pil_to_bgr(src_pil)
        h, w = src_bgr.shape[:2]

        # ✅ Enforce same size constraint
        if (h != target_h) or (w != target_w):
            st.error(
                f"❌ Size mismatch for {f.name}: "
                f"Source={w}x{h}, Target={target_w}x{target_h}. "
                "Please upload a matching-size target background."
            )
            st.stop()

        # 1) Base alpha from U2Net (cached)
        alpha_u2net = get_cached_u2net(file_bytes, src_pil)

        # 2) Refine alpha using GrabCut
        alpha_grabcut = refine_alpha_mask_with_grabcut(
            original_bgr=src_bgr,
            alpha=alpha_u2net,
            feather_px=feather_px,
            dilate_px=dilate_px,
        )

        # 3) OCR text stroke mask (cached)
        ocr_mask = get_cached_ocr(file_bytes, src_bgr)

        # 4) CTA protection mask
        cta_mask = get_cta_saturation_mask(src_bgr)

        # 5) Final foreground = union
        alpha_final = merge_foreground_masks(alpha_grabcut, ocr_mask, cta_mask)

        # 6) Transplant background (no recolor involved)
        out_bgr = transplant_background(
            source_bgr=src_bgr,
            target_bg_bgr=target_bgr,
            alpha_fg=alpha_final,
        )

        out_pil = bgr_to_pil(out_bgr)

        safe_name = f"{f.name.rsplit('.', 1)[0]}_transplanted.png"
        results_for_zip.append((safe_name, out_pil))

        if i < 5:
            c1, c2 = st.columns(2)
            with c1:
                st.image(src_pil, caption=f"Source: {f.name}", use_container_width=True)
            with c2:
                st.image(out_pil, caption=f"Output: {safe_name}", use_container_width=True)

        progress.progress(int(((i + 1) / len(source_files)) * 100))

    st.success("✅ Background transplant complete!")

    zip_bytes = images_to_zip_bytes(results_for_zip)

    st.download_button(
        label="⬇️ Download Output ZIP",
        data=zip_bytes,
        file_name="transplanted_banners.zip",
        mime="application/zip"
    )
