DIGIHAAT BULK BANNER ASSIGNMENT
A Streamlit-based tool that batch-transforms banner background theme colors while preserving all foreground details such as text, CTA buttons, product images, icons, and shadows.
This project is designed for automating bulk creative production workflows where multiple banners need the same theme update without manually editing each file.

>>> Features
- Bulk Upload(PNG / JPG / WEBP)  
- Background Theme Recoloring with gradient preservation  
- Foreground Protection using AI segmentation (U2Net via `rembg`) 
- Adjustable controls:
  -- Recolor Strength (how strong the theme shift is)
  -- Mask Feather (edge smoothing)
  -- Mask Dilation (shadow preservation)
- Preview (Original vs Output side-by-side)  
- Download Output as ZIP (processed banners saved in PNG format)

>>> How it Works (Technical Overview)
1) Foreground/Background Separation
- Generates an alpha matte using U2Net segmentation (rembg).
- Refines mask using OpenCV GrabCut to better preserve UI components (buttons, text blocks).
2) Background Theme Transformation
- Converts only the background pixels
- Preserves gradient/lighting to avoid flat or unnatural results
- Foreground pixels are always taken from the original image to prevent any detail loss
3) Batch Output
- Processes a full batch of uploaded images
- Exports all results into a downloadable .zip

>>> Project Structure
banner/
│
├── app.py
├── requirements.txt
│
└── core/
    ├── __init__.py
    ├── masking.py
    ├── recolor.py
    └── zip_export.py
