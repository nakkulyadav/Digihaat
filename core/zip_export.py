from typing import List, Tuple
from PIL import Image
import io
import zipfile


def images_to_zip_bytes(items: List[Tuple[str, Image.Image]]) -> bytes:
    """
    Creates a ZIP (in-memory) from a list of (filename, PIL.Image).
    Returns zip bytes for Streamlit download_button.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, img in items:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            zf.writestr(name, img_bytes.getvalue())

    buf.seek(0)
    return buf.getvalue()
