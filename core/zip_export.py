import io
import zipfile
from PIL import Image


def images_to_zip_bytes(results):
    """
    results: list of tuples -> (filename, PIL.Image)
    Returns: bytes (zip)
    """
    mem_zip = io.BytesIO()

    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, pil_img in results:
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")  # always PNG for quality
            img_bytes.seek(0)
            zf.writestr(fname, img_bytes.read())

    mem_zip.seek(0)
    return mem_zip.getvalue()
