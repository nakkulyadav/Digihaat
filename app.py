import streamlit as st
import numpy as np
import cv2
from PIL import Image

from core.masking import get_alpha_mask_u2net, refine_alpha_mask
from core.recolor import recolor_background_preserve_gradient
from core.zip_export import images_to_zip_bytes


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


st.set_page_config(page_title="Bulk Banner BG Transformer", layout="wide")
st.title("🎨 Bulk Banner Background Theme Transformer")
st.caption("Change ONLY the background theme color while preserving foreground exactly (text/products/buttons).")

# Sidebar controls
st.sidebar.header("Settings")

target_color = st.sidebar.color_picker("Target Background Theme Color", value="#F3E6D8")
hex_input = st.sidebar.text_input("HEX Input (optional override)", value=target_color)

# Decide final color
final_color = hex_input.strip() if hex_input.strip() else target_color

strength = st.sidebar.slider("Recolor Strength (theme intensity)", 0.0, 1.0, 0.85, 0.01)
feather_px = st.sidebar.slider("Mask Feather (edge smoothing)", 0, 15, 3, 1)
dilate_px = st.sidebar.slider("Mask Dilation (shadow preservation)", 0, 6, 1, 1)

st.sidebar.markdown("---")
st.sidebar.write("✅ Output format: PNG")
st.sidebar.write("✅ Batch download: ZIP")

uploaded_files = st.file_uploader(
    "Upload banners (PNG / JPG / WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload multiple banners to start.")
    st.stop()

st.write(f"✅ {len(uploaded_files)} image(s) uploaded")

process_btn = st.button("🚀 Process Batch", type="primary")

if process_btn:
    results_for_zip = []
    preview_cols = st.columns(2)

    st.subheader("Preview (first few images)")
    progress = st.progress(0)

    for i, f in enumerate(uploaded_files):
        pil_img = Image.open(f).convert("RGB")
        original_bgr = pil_to_bgr(pil_img)

        # 1) Mask
        alpha = get_alpha_mask_u2net(pil_img)
        alpha = refine_alpha_mask(alpha, feather_px=feather_px, dilate_px=dilate_px)

        # 2) Recolor
        out_bgr = recolor_background_preserve_gradient(
            original_bgr=original_bgr,
            alpha_fg=alpha,
            target_hex=final_color,
            strength=strength
        )

        out_pil = bgr_to_pil(out_bgr)

        # Save for ZIP
        safe_name = f"{f.name.rsplit('.', 1)[0]}_bg.png"
        results_for_zip.append((safe_name, out_pil))

        # Preview first 5
        if i < 5:
            with st.container():
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
