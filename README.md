# Car Door Fabric Defect Detection (Streamlit + OpenCV)

This app performs texture segmentation using a Gabor filter bank (Jain & Farrokhnia, 1991) to detect defects in car door fabric. Upload an image, segment textures with KMeans, and highlight defects as minority texture regions.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Then upload a fabric image (PNG/JPG) via the UI.

## Features

- Image uploader with on-the-fly processing
- Gabor filter bank across multiple wavelengths and orientations
- Gaussian smoothing matched to each wavelength (`K` factor)
- Spatial features (`X`, `Y`) added to Gabor magnitudes
- KMeans clustering (2–4 clusters) for texture segmentation
- Auto/manual selection of the defect cluster
- Overlay visualization and downloadable defect mask/overlay
- Optional saving to `output/`
 - Optional PCA visualization of the feature space (1st component)
 - Adjustable Gabor `gamma` (aspect ratio) to tune anisotropy

## Sidebar Settings

- Resize scale: downsample for speed and stability
- Orientation step: 15/30/45/60 degrees
- Smoothing factor `K`: controls Gaussian smoothing strength
- Number of clusters: typically 2 for fabric vs defect; increase if needed
- Defect selection: auto (minority cluster) or manual label
 - Gabor gamma: controls filter aspect ratio (anisotropy)
 - PCA feature map: visualize separability of features

## Outputs

When "Save outputs to output/" is enabled, the app writes:
- `<name>_seg.png`: colored segmentation labels
- `<name>_overlay.png`: defect overlay on the original
- `<name>_mask.png`: binary defect mask (white = defect)

You can also download the mask and overlay directly from the UI.

## Notes for Car Door Fabric

- Ensure consistent lighting and minimal glare for best results.
- If the fabric pattern is highly directional, try smaller orientation steps (e.g., 30°) to better capture anisotropic textures.
- If normal fabric dominates and defects are small, "auto" mode usually works well since it selects the minority cluster as defect.


