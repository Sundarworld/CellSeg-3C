# pap_ui.py
# Streamlit UI for Pap cell feature extraction (20 features + Class)
# Labels (exactly as in martin2003):
# - Background: 1 or 4
# - Nucleus (Kerne): 2
# - Cytoplasm (Cyto): 0 or 3

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from skimage import measure
from skimage.feature import peak_local_max

# ---------------------------------------------
# UI CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Pap Cell Feature Extractor (20 Features + Class)", layout="wide")

# ---------------------------------------------
# CONSTANTS — label scheme
# ---------------------------------------------
NUCLEUS_LABEL = 2
CYTO_LABELS = {0, 3}
BACKGROUND_LABELS = {1, 4}

# ---------------------------------------------
# HELPERS
# ---------------------------------------------
def _load_image(file) -> Image.Image:
    return Image.open(file)

def _ensure_uint8_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr

def _validate_labels(mask: np.ndarray) -> Tuple[bool, str]:
    unique_vals = set(np.unique(mask).tolist())
    allowed = {0, 1, 2, 3, 4}
    extra = unique_vals - allowed
    if extra:
        return False, f"Mask contains unexpected labels {sorted(extra)}. Expected {allowed}"
    if NUCLEUS_LABEL not in unique_vals:
        return False, "Mask missing nucleus pixels (2)."
    if not (unique_vals & CYTO_LABELS):
        return False, "Mask missing cytoplasm pixels (0 or 3)."
    return True, "OK"

def _rgb_to_luma_Y(orig_img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    rgb = orig_img.convert("RGB").resize(size)
    arr = np.asarray(rgb).astype(np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

def _areas_from_mask(mask: np.ndarray, pixel_size_um: Optional[float]) -> Tuple[float, float, str]:
    nucleus_pixels = np.count_nonzero(mask == NUCLEUS_LABEL)
    cyto_pixels = np.count_nonzero(np.isin(mask, list(CYTO_LABELS)))
    if pixel_size_um and pixel_size_um > 0:
        factor = pixel_size_um ** 2
        return nucleus_pixels * factor, cyto_pixels * factor, "µm²"
    else:
        return float(nucleus_pixels), float(cyto_pixels), "pixels"

def _perimeter_len(binary_mask: np.ndarray) -> float:
    try:
        return float(measure.perimeter(binary_mask, neighbourhood=8))
    except TypeError:
        try:
            return float(measure.perimeter(binary_mask, neighborhood=8))
        except TypeError:
            return float(measure.perimeter_crofton(binary_mask, directions=4))

def _regionprops_metrics(binary_mask: np.ndarray, pixel_size_um: Optional[float]) -> Dict[str, float]:
    if binary_mask.sum() == 0:
        return {k: float("nan") for k in ["area","perimeter","major_axis","minor_axis","elongation","roundness"]}
    labeled = measure.label(binary_mask, connectivity=2)
    reg = max(measure.regionprops(labeled), key=lambda r: r.area)
    s_lin = pixel_size_um if (pixel_size_um and pixel_size_um > 0) else 1.0
    area = float(reg.area) * (s_lin ** 2)
    perim = _perimeter_len(binary_mask) * s_lin
    major = float(reg.major_axis_length) * s_lin
    minor = float(reg.minor_axis_length) * s_lin
    elong = float(major / minor) if minor > 0 else float("nan")
    roundness = float((4.0 * np.pi * area) / (perim ** 2)) if perim > 0 else float("nan")
    return {"area": area,"perimeter": perim,"major_axis": major,"minor_axis": minor,"elongation": elong,"roundness": roundness}

def _centroid_offset_norm(binary_mask: np.ndarray) -> float:
    if binary_mask.sum() == 0: return float("nan")
    reg = max(measure.regionprops(measure.label(binary_mask, connectivity=2)), key=lambda r: r.area)
    cy, cx = reg.centroid
    h, w = binary_mask.shape
    dist = np.hypot(cy - h/2, cx - w/2)
    return float(dist / np.hypot(h/2, w/2))

def _count_local_extrema(intensity: np.ndarray, roi_mask: np.ndarray) -> Tuple[int,int]:
    if roi_mask.sum() == 0: return 0,0
    maxima = peak_local_max(np.where(roi_mask, intensity, -np.inf), min_distance=2, threshold_rel=0.05, exclude_border=True)
    minima = peak_local_max(np.where(roi_mask, -intensity, -np.inf), min_distance=2, threshold_rel=0.05, exclude_border=True)
    return int(maxima.shape[0]), int(minima.shape[0])

def _pseudo_color_mask(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[np.isin(mask, list(BACKGROUND_LABELS))] = (32,32,32)
    rgb[np.isin(mask, list(CYTO_LABELS))] = (0,180,0)
    rgb[mask==NUCLEUS_LABEL] = (220,20,60)
    return Image.fromarray(rgb, "RGB")

def _overlay_on_original(original: Image.Image, mask_vis: Image.Image, alpha=0.45) -> Image.Image:
    return Image.blend(original.convert("RGB").resize(mask_vis.size), mask_vis, alpha)

# ---------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------
def compute_full_features(mask_arr: np.ndarray, original_img: Optional[Image.Image], pixel_size_um: Optional[float], class_id: int) -> Dict[str, float]:
    nuc_mask = (mask_arr == NUCLEUS_LABEL)
    cyto_mask = np.isin(mask_arr, list(CYTO_LABELS))

    # ✅ FIXED: pass pixel_size_um directly; let _areas_from_mask handle None/0
    kerne_A, cyto_A, _ = _areas_from_mask(mask_arr, pixel_size_um)

    nuc_m = _regionprops_metrics(nuc_mask, pixel_size_um)
    cyt_m = _regionprops_metrics(cyto_mask, pixel_size_um)

    if original_img is not None:
        Y = _rgb_to_luma_Y(original_img, size=mask_arr.shape[::-1])
        kerneY = float(Y[nuc_mask].mean()) if nuc_mask.any() else np.nan
        cytoY = float(Y[cyto_mask].mean()) if cyto_mask.any() else np.nan
        kMax, kMin = _count_local_extrema(Y, nuc_mask)
        cMax, cMin = _count_local_extrema(Y, cyto_mask)
    else:
        kerneY = cytoY = np.nan
        kMax = kMin = cMax = cMin = 0

    return {
        "Kerne_A": float(kerne_A),
        "Cyto_A": float(cyto_A),
        "K/C": float(kerne_A/cyto_A) if cyto_A>0 else np.nan,
        "Kerne_Ycol": kerneY,
        "Cyto_Ycol": cytoY,
        "KerneShort": nuc_m["minor_axis"],
        "KerneLong": nuc_m["major_axis"],
        "KerneElong": nuc_m["elongation"],
        "KerneRund": nuc_m["roundness"],
        "CytoShort": cyt_m["minor_axis"],
        "CytoLong": cyt_m["major_axis"],
        "CytoElong": cyt_m["elongation"],
        "CytoRund": cyt_m["roundness"],
        "KernePeri": nuc_m["perimeter"],
        "CytoPeri": cyt_m["perimeter"],
        "KernePos": _centroid_offset_norm(nuc_mask),
        "KerneMax": kMax,
        "KerneMin": kMin,
        "CytoMax": cMax,
        "CytoMin": cMin,
        "Class": class_id
    }

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
with st.sidebar:
    st.markdown("## Settings")
    pixel_size_um = st.number_input("Pixel size (µm/pixel). Leave 0 for raw pixels:", min_value=0.0, value=0.0, step=0.01)
    class_id = st.selectbox("Class label (1–7):", options=list(range(1,8)), index=0)
    st.markdown("---")
    st.markdown("""
**Label scheme (fixed):**

- Background: 1 or 4  
- Nucleus (Kerne): 2  
- Cytoplasm (Cyto): 0 or 3
""")

# ---------------------------------------------
# MAIN UI
# ---------------------------------------------
st.title("CellSeg-3C ")

left, right = st.columns(2)
with left:
    original_file = st.file_uploader("Original image (optional, for intensity/texture)", type=["bmp","png","jpg","jpeg","tif","tiff"], key="orig")
    mask_file = st.file_uploader("Segmented/derived mask (required)", type=["bmp","png","tif","tiff"], key="mask")

with right:
    if mask_file is not None:
        mask_img = _ensure_uint8_mask(_load_image(mask_file))
        ok,msg = _validate_labels(mask_img)
        if original_file is not None:
            overlay = _overlay_on_original(_load_image(original_file), mask_vis)
            st.image(overlay, caption="Overlay on original")
        if not ok: st.error(msg); st.stop()
        mask_vis = _pseudo_color_mask(mask_img)
        st.image(mask_vis, caption="Segmentation (pseudo-color)")

    else:
        st.info("Upload the segmented/derived mask to begin.")

st.markdown("---")

if mask_file is not None:
    mask_arr = _ensure_uint8_mask(_load_image(mask_file))
    orig_img = _load_image(original_file) if original_file else None
    feats = compute_full_features(mask_arr, orig_img, pixel_size_um if pixel_size_um>0 else None, class_id)
    image_name = getattr(mask_file, "name", "uploaded_mask")
    df = pd.DataFrame([{**{"Image name": image_name}, **feats}])
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"features_{image_name}.csv", mime="text/csv")
else:
    st.stop()
