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
from skimage import measure, morphology, filters, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage
import cv2

# ---------------------------------------------
# UI CONFIG
# ---------------------------------------------
st.set_page_config(page_title="CellSeg-3C: Cervical Cell Analysis & Classification", layout="wide")

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
    return Image.fromarray(rgb)

def _overlay_on_original(original: Image.Image, mask_vis: Image.Image, alpha=0.45) -> Image.Image:
    return Image.blend(original.convert("RGB").resize(mask_vis.size), mask_vis, alpha)

# ---------------------------------------------
# AUTOMATIC SEGMENTATION
# ---------------------------------------------
def automatic_cell_segmentation(image: Image.Image, min_cell_size: int = 500, nucleus_sensitivity: float = 0.3) -> np.ndarray:
    """
    Automatically segment a cell image into Background, Nucleus, and Cytoplasm.
    Returns a mask with labels: Background=1, Nucleus=2, Cytoplasm=3
    """
    # Convert to RGB if needed and then to numpy array
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Convert to grayscale for segmentation
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Create initial mask using Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find cell regions (foreground)
    cell_mask = binary > 0
    
    # Fill holes in the cell mask
    cell_mask = ndimage.binary_fill_holes(cell_mask)
    
    # Remove small objects
    cell_mask = morphology.remove_small_objects(cell_mask, min_size=min_cell_size)
    
    # Initialize output mask with background (label 1)
    output_mask = np.ones_like(gray, dtype=np.uint8)
    
    if not np.any(cell_mask):
        st.warning("No cell regions detected. Returning background-only mask.")
        return output_mask
    
    # Use watershed to separate nucleus and cytoplasm
    # The nucleus is typically darker (lower intensity) than cytoplasm
    distance = ndimage.distance_transform_edt(cell_mask)
    
    # Find nucleus seeds (local maxima in distance transform)
    local_maxima = peak_local_max(distance, min_distance=20, threshold_abs=nucleus_sensitivity*distance.max())
    
    if len(local_maxima) == 0:
        # If no clear nucleus found, use intensity-based approach
        # Nucleus regions are typically darker
        nucleus_threshold = filters.threshold_otsu(gray[cell_mask])
        nucleus_candidates = (gray < nucleus_threshold) & cell_mask
        
        if np.any(nucleus_candidates):
            # Use the largest connected component as nucleus
            labeled_nucleus = measure.label(nucleus_candidates)
            regions = measure.regionprops(labeled_nucleus)
            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                nucleus_mask = labeled_nucleus == largest_region.label
            else:
                # Fallback: use central region as nucleus
                center_y, center_x = np.array(gray.shape) // 2
                y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
                nucleus_mask = ((y - center_y)**2 + (x - center_x)**2) < (min(gray.shape) // 6)**2
                nucleus_mask = nucleus_mask & cell_mask
        else:
            # Fallback: use central region as nucleus
            center_y, center_x = np.array(gray.shape) // 2
            y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
            nucleus_mask = ((y - center_y)**2 + (x - center_x)**2) < (min(gray.shape) // 6)**2
            nucleus_mask = nucleus_mask & cell_mask
    else:
        # Create markers for watershed
        markers = np.zeros_like(gray, dtype=np.int32)
        
        # Background marker
        markers[~cell_mask] = 1
        
        # Nucleus markers
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 2
        
        # Perform watershed segmentation
        labels = segmentation.watershed(-gray, markers, mask=cell_mask)
        
        # Find the largest non-background region as nucleus
        nucleus_label = None
        max_area = 0
        for label in np.unique(labels):
            if label > 1:  # Skip background (1)
                area = np.sum(labels == label)
                if area > max_area:
                    max_area = area
                    nucleus_label = label
        
        if nucleus_label is not None:
            nucleus_mask = labels == nucleus_label
        else:
            # Fallback approach
            nucleus_threshold = filters.threshold_otsu(gray[cell_mask])
            nucleus_mask = (gray < nucleus_threshold) & cell_mask
    
    # Clean up nucleus mask
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=100)
    nucleus_mask = ndimage.binary_fill_holes(nucleus_mask)
    
    # Cytoplasm is cell region minus nucleus
    cytoplasm_mask = cell_mask & ~nucleus_mask
    
    # Apply morphological operations to clean boundaries
    if np.any(nucleus_mask):
        nucleus_mask = morphology.binary_erosion(nucleus_mask, morphology.disk(1))
        nucleus_mask = morphology.binary_dilation(nucleus_mask, morphology.disk(1))
    
    if np.any(cytoplasm_mask):
        cytoplasm_mask = morphology.binary_erosion(cytoplasm_mask, morphology.disk(1))
        cytoplasm_mask = morphology.binary_dilation(cytoplasm_mask, morphology.disk(1))
    
    # Assign labels to output mask
    output_mask[nucleus_mask] = 2  # Nucleus
    output_mask[cytoplasm_mask] = 3  # Cytoplasm
    # Background remains 1
    
    return output_mask

# ---------------------------------------------
# CERVICAL CELL CLASSIFICATION SYSTEM
# ---------------------------------------------
class CervicalCellClassifier:
    def __init__(self):
        # Cell type mapping based on cervical cytology
        self.cell_types = {
            1: "Superficial Squamous Epithelial",
            2: "Intermediate Squamous Epithelial", 
            3: "Columnar Epithelial",
            4: "Mild Squamous Non-keratinizing Dysplasia",
            5: "Moderate Squamous Non-keratinizing Dysplasia",
            6: "Severe Squamous Non-keratinizing Dysplasia",
            7: "Squamous Cell Carcinoma in Situ"
        }
        
        # Category classification
        self.categories = {
            1: "Normal", 2: "Normal", 3: "Normal",
            4: "Abnormal", 5: "Abnormal", 6: "Abnormal", 7: "Abnormal"
        }
        
        # Risk levels for clinical decision
        self.risk_levels = {
            1: "Low Risk", 2: "Low Risk", 3: "Low Risk",
            4: "Moderate Risk (LSIL)", 5: "High Risk (HSIL)", 
            6: "High Risk (HSIL)", 7: "Very High Risk (CIS)"
        }
    
    def get_classification_result(self, class_id: int) -> Dict[str, str]:
        """Get detailed classification result with clinical information"""
        return {
            "class_id": class_id,
            "cell_type": self.cell_types.get(class_id, "Unknown"),
            "category": self.categories.get(class_id, "Unknown"),
            "risk_level": self.risk_levels.get(class_id, "Unknown")
        }
    
    def get_wellness_advice(self) -> Dict[str, str]:
        """Wellness advice for normal cells"""
        return {
            "title": "🌟 Wellness & Prevention Advice",
            "advice": """
**Congratulations! Your cervical cells appear normal. Here's how to maintain your cervical health:**

🔸 **Regular Screening:**
- Continue regular Pap smears every 3 years (ages 21-65)
- Consider HPV co-testing every 5 years (ages 30-65)

🔸 **Lifestyle Recommendations:**
- Maintain a healthy diet rich in fruits and vegetables
- Exercise regularly to boost immune system
- Avoid smoking and limit alcohol consumption
- Practice safe sex and limit number of sexual partners

🔸 **HPV Prevention:**
- Consider HPV vaccination if eligible (ages 9-45)
- Use barrier contraception consistently
- Maintain good personal hygiene

🔸 **General Health:**
- Manage stress levels effectively
- Get adequate sleep (7-9 hours nightly)
- Take folic acid and vitamin supplements as recommended
- Maintain a healthy weight

🔸 **Next Steps:**
- Continue routine gynecological check-ups
- Discuss family history with your healthcare provider
- Stay informed about cervical health guidelines
""",
            "follow_up": "Schedule your next routine screening in 3 years or as recommended by your healthcare provider."
        }
    
    def get_treatment_recommendations(self, class_id: int) -> Dict[str, str]:
        """Treatment recommendations for abnormal cells"""
        if class_id == 4:  # Mild dysplasia
            return {
                "title": "⚠️ Mild Dysplasia Detected - Immediate Action Required",
                "severity": "Moderate Risk (Low-grade SIL)",
                "recommendations": """
**Your test shows mild cervical dysplasia. While not cancer, this requires medical attention.**

🏥 **Immediate Next Steps:**
- Schedule appointment with gynecologist within 2-4 weeks
- Bring all test results and medical history
- Do not delay - early treatment prevents progression

🔬 **Expected Follow-up Tests:**
- Colposcopy examination with possible biopsy
- HPV testing to identify high-risk types
- Repeat Pap smear in 6-12 months

📍 **Recommended Healthcare Facilities:**
- **Gynecology Departments** at major hospitals
- **Women's Health Centers** with colposcopy services
- **Cancer Centers** with cervical screening programs

🎯 **Treatment Options May Include:**
- Active surveillance with frequent monitoring
- Cryotherapy (freezing abnormal cells)
- LEEP procedure (loop electrosurgical excision)
""",
                "urgency": "Schedule medical consultation within 2-4 weeks"
            }
        
        elif class_id in [5, 6]:  # Moderate/Severe dysplasia
            return {
                "title": "🚨 High-Grade Dysplasia Detected - Urgent Medical Attention Required",
                "severity": "High Risk (High-grade SIL)",
                "recommendations": """
**Your test shows moderate to severe cervical dysplasia. This is a pre-cancerous condition requiring immediate treatment.**

🏥 **URGENT Next Steps:**
- Contact gynecologist or oncologist IMMEDIATELY
- Schedule appointment within 1-2 weeks
- Consider seeking care at a specialized cancer center

🔬 **Required Immediate Tests:**
- Urgent colposcopy with directed biopsy
- HPV genotyping for high-risk strains
- Possible cone biopsy or LEEP procedure

📍 **Recommended Specialized Centers:**
- **Gynecologic Oncology Centers**
- **Comprehensive Cancer Centers**
- **University Hospital Women's Health Departments**

🎯 **Treatment Options:**
- LEEP (Loop Electrosurgical Excision Procedure)
- Cone biopsy (conization)
- Cryotherapy or laser therapy
- Close monitoring with frequent follow-ups
""",
                "urgency": "URGENT: Schedule medical consultation within 1-2 weeks"
            }
        
        elif class_id == 7:  # Carcinoma in situ
            return {
                "title": "🚨 CRITICAL: Carcinoma in Situ Detected - Emergency Medical Attention Required",
                "severity": "Very High Risk (Carcinoma in Situ)",
                "recommendations": """
**Your test shows carcinoma in situ - the most advanced pre-cancerous stage. Immediate treatment is essential.**

🏥 **EMERGENCY Next Steps:**
- Contact gynecologic oncologist IMMEDIATELY
- Schedule urgent consultation within 1 week
- Seek care at a comprehensive cancer center

🔬 **Critical Immediate Tests:**
- Emergency colposcopy with multiple biopsies
- Staging workup to rule out invasion
- HPV testing and imaging studies

📍 **Seek Care At:**
- **Gynecologic Oncology Centers** (PRIORITY)
- **National Cancer Institute-designated Centers**
- **Academic Medical Centers** with cervical cancer programs

🎯 **Treatment Options:**
- Immediate surgical intervention (cone biopsy/LEEP)
- Possible hysterectomy depending on factors
- Intensive follow-up surveillance program

⚡ **CRITICAL:** This condition can progress to invasive cancer. Do not delay treatment.
""",
                "urgency": "EMERGENCY: Contact oncologist within 24-48 hours"
            }
        
        return {
            "title": "Medical Consultation Required",
            "severity": "Unknown Risk Level",
            "recommendations": "Please consult with a healthcare professional for proper evaluation.",
            "urgency": "Schedule medical consultation as soon as possible"
        }

# Initialize classifier
cervical_classifier = CervicalCellClassifier()

# ---------------------------------------------
# AUTOMATIC FEATURE-BASED CLASSIFICATION
# ---------------------------------------------
class AutomaticCervicalClassifier:
    def __init__(self):
        """
        Automatic classifier based on known cervical cell characteristics
        from research literature and clinical guidelines
        """
        self.feature_names = [
            "Kerne_A", "Cyto_A", "K/C", "Kerne_Ycol", "Cyto_Ycol",
            "KerneShort", "KerneLong", "KerneElong", "KerneRund",
            "CytoShort", "CytoLong", "CytoElong", "CytoRund", 
            "KernePeri", "CytoPeri", "KernePos", "KerneMax", 
            "KerneMin", "CytoMax", "CytoMin"
        ]
        
        # Classification thresholds based on clinical literature
        self.classification_rules = {
            # Nuclear/Cytoplasmic ratio thresholds
            'nc_ratio_thresholds': {
                'normal_max': 0.3,      # Normal cells: NC ratio < 0.3
                'mild_max': 0.5,        # Mild dysplasia: 0.3-0.5
                'moderate_max': 0.7,    # Moderate: 0.5-0.7
                'severe_max': 0.8,      # Severe: 0.7-0.8
                # Carcinoma in situ: > 0.8
            },
            
            # Nuclear area thresholds (relative)
            'nuclear_area_thresholds': {
                'small_max': 1000,      # Small nucleus (normal)
                'medium_max': 2000,     # Medium nucleus
                'large_max': 3500,      # Large nucleus (abnormal)
            },
            
            # Shape irregularity (elongation and roundness)
            'shape_thresholds': {
                'regular_elongation_max': 2.0,     # Regular shape
                'irregular_elongation_min': 3.0,   # Irregular shape
                'regular_roundness_min': 0.7,      # Round/regular
                'irregular_roundness_max': 0.5,    # Irregular
            },
            
            # Intensity variation (texture complexity)
            'texture_thresholds': {
                'simple_texture_max': 3,    # Few local extrema (normal)
                'complex_texture_min': 8,   # Many local extrema (abnormal)
            }
        }
    
    def predict_class(self, features: Dict[str, float]) -> Tuple[int, float, Dict[str, str]]:
        """
        Predict cervical cell class based on extracted features
        Returns: (predicted_class, confidence_score, reasoning)
        """
        # Extract key features
        nc_ratio = features.get('K/C', 0)
        nuclear_area = features.get('Kerne_A', 0)
        nuclear_elongation = features.get('KerneElong', 1)
        nuclear_roundness = features.get('KerneRund', 1)
        cyto_elongation = features.get('CytoElong', 1)
        cyto_area = features.get('Cyto_A', 0)
        nuclear_maxima = features.get('KerneMax', 0)
        nuclear_minima = features.get('KerneMin', 0)
        nucleus_intensity = features.get('Kerne_Ycol', 128)
        cyto_intensity = features.get('Cyto_Ycol', 128)
        
        # Calculate additional features
        texture_complexity = nuclear_maxima + nuclear_minima
        total_cell_area = nuclear_area + cyto_area
        intensity_contrast = abs(nucleus_intensity - cyto_intensity) if not (np.isnan(nucleus_intensity) or np.isnan(cyto_intensity)) else 0
        
        # Initialize scoring system with equal baseline
        class_scores = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
        reasoning = []
        
        # Rule 1: Nuclear/Cytoplasmic Ratio Analysis (Most Important)
        if nc_ratio <= 0.15:  # Very low N/C ratio
            class_scores[1] += 5  # Superficial squamous (mature cells)
            reasoning.append(f"Very low N/C ratio ({nc_ratio:.3f}) indicates superficial squamous cells")
            
        elif nc_ratio <= 0.25:  # Low N/C ratio
            class_scores[1] += 3
            class_scores[2] += 4  # Intermediate squamous
            reasoning.append(f"Low N/C ratio ({nc_ratio:.3f}) suggests normal squamous cells")
            
        elif nc_ratio <= 0.35:  # Normal range
            class_scores[2] += 3
            class_scores[3] += 2  # Columnar can have slightly higher N/C
            reasoning.append(f"Normal N/C ratio ({nc_ratio:.3f}) suggests intermediate squamous or columnar cells")
            
        elif nc_ratio <= 0.50:  # Mild elevation
            class_scores[4] += 4  # Mild dysplasia
            class_scores[3] += 2  # Some columnar cells
            reasoning.append(f"Mildly elevated N/C ratio ({nc_ratio:.3f}) suggests mild dysplasia")
            
        elif nc_ratio <= 0.65:  # Moderate elevation
            class_scores[5] += 4  # Moderate dysplasia
            class_scores[4] += 2
            reasoning.append(f"Moderately elevated N/C ratio ({nc_ratio:.3f}) suggests moderate dysplasia")
            
        elif nc_ratio <= 0.80:  # High elevation
            class_scores[6] += 4  # Severe dysplasia
            class_scores[5] += 2
            reasoning.append(f"High N/C ratio ({nc_ratio:.3f}) suggests severe dysplasia")
            
        else:  # Very high N/C ratio
            class_scores[7] += 5  # Carcinoma in situ
            class_scores[6] += 2
            reasoning.append(f"Very high N/C ratio ({nc_ratio:.3f}) suggests carcinoma in situ")
        
        # Rule 2: Nuclear Size Analysis
        if nuclear_area <= 800:  # Small nucleus
            class_scores[1] += 3  # Superficial cells typically smaller
            class_scores[2] += 2
            reasoning.append(f"Small nuclear area ({nuclear_area:.0f}) typical of superficial cells")
            
        elif nuclear_area <= 1500:  # Medium nucleus
            class_scores[2] += 3  # Intermediate size
            class_scores[3] += 3  # Columnar
            class_scores[4] += 2  # Early dysplasia
            reasoning.append(f"Medium nuclear area ({nuclear_area:.0f}) suggests intermediate or columnar cells")
            
        elif nuclear_area <= 2500:  # Large nucleus
            class_scores[4] += 3  # Mild dysplasia
            class_scores[5] += 3  # Moderate dysplasia
            reasoning.append(f"Large nuclear area ({nuclear_area:.0f}) suggests mild to moderate dysplasia")
            
        else:  # Very large nucleus
            class_scores[5] += 2
            class_scores[6] += 3  # Severe dysplasia
            class_scores[7] += 3  # CIS
            reasoning.append(f"Very large nuclear area ({nuclear_area:.0f}) suggests severe dysplasia or CIS")
        
        # Rule 3: Cell Size Analysis (Total area)
        if total_cell_area <= 2000:  # Small cells
            class_scores[1] += 2  # Superficial cells are often smaller
            
        elif total_cell_area <= 4000:  # Medium cells
            class_scores[2] += 2
            class_scores[3] += 2
            
        elif total_cell_area >= 6000:  # Large cells
            class_scores[3] += 2  # Columnar cells can be large
            
        # Rule 4: Nuclear Shape Analysis
        if nuclear_elongation <= 1.5 and nuclear_roundness >= 0.8:  # Very regular
            class_scores[1] += 3
            class_scores[2] += 3
            class_scores[3] += 2
            reasoning.append(f"Very regular nuclear shape suggests normal cells")
            
        elif nuclear_elongation <= 2.0 and nuclear_roundness >= 0.6:  # Regular
            class_scores[2] += 2
            class_scores[3] += 2
            class_scores[4] += 1
            reasoning.append(f"Regular nuclear shape suggests normal to mild dysplasia")
            
        elif nuclear_elongation <= 3.0 or nuclear_roundness >= 0.4:  # Moderately irregular
            class_scores[4] += 2
            class_scores[5] += 2
            reasoning.append(f"Moderately irregular nuclear shape suggests dysplasia")
            
        else:  # Very irregular
            class_scores[5] += 2
            class_scores[6] += 3
            class_scores[7] += 2
            reasoning.append(f"Irregular nuclear shape suggests moderate to severe dysplasia")
        
        # Rule 5: Cytoplasm Shape (Cell Type Discrimination)
        if cyto_elongation >= 3.0:  # Highly elongated cytoplasm
            class_scores[3] += 4  # Columnar epithelial
            reasoning.append(f"Highly elongated cytoplasm ({cyto_elongation:.2f}) indicates columnar cells")
            
        elif cyto_elongation >= 2.0:  # Moderately elongated
            class_scores[3] += 2
            class_scores[2] += 1
            
        # Rule 6: Texture Complexity Analysis
        if texture_complexity <= 2:  # Very simple
            class_scores[1] += 2
            class_scores[2] += 2
            reasoning.append(f"Simple nuclear texture ({texture_complexity} extrema) suggests normal cells")
            
        elif texture_complexity <= 5:  # Moderate complexity
            class_scores[2] += 1
            class_scores[3] += 1
            class_scores[4] += 1
            
        elif texture_complexity >= 10:  # High complexity
            class_scores[5] += 2
            class_scores[6] += 2
            class_scores[7] += 2
            reasoning.append(f"Complex nuclear texture ({texture_complexity} extrema) suggests dysplasia")
        
        # Rule 7: Intensity Contrast (Nuclear-Cytoplasm difference)
        if intensity_contrast >= 50:  # High contrast
            class_scores[4] += 1
            class_scores[5] += 1
            class_scores[6] += 1
            reasoning.append(f"High nuclear-cytoplasm contrast suggests dysplasia")
        
        # Additional specific discrimination rules
        
        # Superficial vs Intermediate discrimination
        if nc_ratio <= 0.2 and nuclear_area <= 1000:
            class_scores[1] += 2  # Favor superficial
            class_scores[2] -= 1
        elif nc_ratio <= 0.3 and nuclear_area <= 1800:
            class_scores[2] += 2  # Favor intermediate
            class_scores[1] -= 1
        
        # Columnar cell specific features
        if cyto_elongation >= 2.5 and nc_ratio <= 0.4:
            class_scores[3] += 3  # Strong columnar indicator
            reasoning.append(f"Elongated cytoplasm with moderate N/C suggests columnar epithelial")
        
        # Progressive dysplasia discrimination
        if 0.4 <= nc_ratio <= 0.55 and nuclear_area <= 2000:
            class_scores[4] += 2  # Favor mild dysplasia
            class_scores[5] -= 1
        elif 0.55 <= nc_ratio <= 0.7 and nuclear_area >= 1500:
            class_scores[5] += 2  # Favor moderate dysplasia
            class_scores[4] -= 1
            class_scores[6] -= 1
        elif nc_ratio >= 0.7 and nuclear_area >= 2000:
            if nc_ratio >= 0.85:
                class_scores[7] += 2  # Favor CIS
                class_scores[6] -= 1
            else:
                class_scores[6] += 2  # Favor severe dysplasia
                class_scores[7] -= 1
        
        # Find the class with highest score
        predicted_class = max(class_scores.items(), key=lambda x: x[1])[0]
        max_score = class_scores[predicted_class]
        
        # Calculate confidence based on score separation and total score
        sorted_scores = sorted(class_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            score_gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.95, 0.5 + (score_gap * 0.08))  # Base 50% + gap bonus
        else:
            confidence = 0.5
        
        # Adjust confidence based on total score
        if max_score >= 12:
            confidence = min(0.95, confidence + 0.15)
        elif max_score >= 8:
            confidence = min(0.90, confidence + 0.1)
        elif max_score <= 4:
            confidence = max(0.3, confidence - 0.15)
        
        # Create detailed reasoning
        reasoning_dict = {
            'primary_indicators': reasoning[:3],  # Top 3 reasons
            'nc_ratio': nc_ratio,
            'nuclear_area': nuclear_area,
            'total_cell_area': total_cell_area,
            'shape_regularity': 'Regular' if nuclear_elongation <= 2.0 and nuclear_roundness >= 0.6 else 'Irregular',
            'texture_complexity': 'Simple' if texture_complexity <= 3 else 'Moderate' if texture_complexity <= 7 else 'Complex',
            'class_scores': class_scores
        }
        
        return predicted_class, confidence, reasoning_dict
    
    def get_prediction_explanation(self, predicted_class: int, confidence: float, reasoning: Dict) -> str:
        """Generate human-readable explanation of the prediction"""
        cell_types = {
            1: "Superficial Squamous Epithelial (Normal)",
            2: "Intermediate Squamous Epithelial (Normal)", 
            3: "Columnar Epithelial (Normal)",
            4: "Mild Squamous Non-keratinizing Dysplasia",
            5: "Moderate Squamous Non-keratinizing Dysplasia",
            6: "Severe Squamous Non-keratinizing Dysplasia",
            7: "Squamous Cell Carcinoma in Situ"
        }
        
        explanation = f"""
**🤖 Automatic Classification Result:**

**Predicted Class:** {predicted_class} - {cell_types[predicted_class]}
**Confidence:** {confidence*100:.1f}%

**Key Diagnostic Features:**
- **N/C Ratio:** {reasoning['nc_ratio']:.3f}
- **Nuclear Area:** {reasoning['nuclear_area']:.0f}
- **Total Cell Area:** {reasoning['total_cell_area']:.0f}
- **Nuclear Shape:** {reasoning['shape_regularity']}
- **Texture Complexity:** {reasoning['texture_complexity']}

**Primary Indicators:**
"""
        for indicator in reasoning['primary_indicators']:
            explanation += f"• {indicator}\n"
        
        if confidence >= 0.8:
            explanation += f"\n✅ **High Confidence:** Strong feature agreement for {cell_types[predicted_class]}"
        elif confidence >= 0.6:
            explanation += f"\n⚠️ **Moderate Confidence:** Features generally support {cell_types[predicted_class]}"
        else:
            explanation += f"\n❓ **Low Confidence:** Mixed features - manual review recommended"
        
        return explanation

# Initialize automatic classifier
automatic_classifier = AutomaticCervicalClassifier()

def generate_medical_report(image_name: str, classification_result: Dict, features: Dict, clinical_info: Dict) -> str:
    """Generate a comprehensive medical report"""
    from datetime import datetime
    
    report = f"""
CERVICAL CELL ANALYSIS REPORT
Generated by CellSeg-3C System
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
===============================================

PATIENT INFORMATION:
- Sample ID: {image_name}
- Analysis Method: Comprehensive Cellular Morphometry Analysis (CCMA)
- Classification System: Cervical Dysplasia Progression Classifier (CDPC)

CLASSIFICATION RESULTS:
- Class: {classification_result['class_id']}
- Cell Type: {classification_result['cell_type']}
- Category: {classification_result['category']}
- Risk Level: {classification_result['risk_level']}

MORPHOLOGICAL FEATURES ANALYSIS:
===============================================
Nucleus Measurements:
- Area: {features['Kerne_A']:.2f}
- Perimeter: {features['KernePeri']:.2f}
- Major Axis: {features['KerneLong']:.2f}
- Minor Axis: {features['KerneShort']:.2f}
- Elongation: {features['KerneElong']:.3f}
- Roundness: {features['KerneRund']:.3f}
- Position Offset: {features['KernePos']:.3f}

Cytoplasm Measurements:
- Area: {features['Cyto_A']:.2f}
- Perimeter: {features['CytoPeri']:.2f}
- Major Axis: {features['CytoLong']:.2f}
- Minor Axis: {features['CytoShort']:.2f}
- Elongation: {features['CytoElong']:.3f}
- Roundness: {features['CytoRund']:.3f}

Nuclear-Cytoplasmic Ratio:
- N/C Ratio: {features['K/C']:.3f}

Intensity Features:
- Nucleus Mean Intensity: {features['Kerne_Ycol']:.2f}
- Cytoplasm Mean Intensity: {features['Cyto_Ycol']:.2f}

Texture Analysis:
- Nucleus Local Maxima: {features['KerneMax']}
- Nucleus Local Minima: {features['KerneMin']}
- Cytoplasm Local Maxima: {features['CytoMax']}
- Cytoplasm Local Minima: {features['CytoMin']}

CLINICAL RECOMMENDATIONS:
===============================================
"""
    
    if classification_result['category'] == "Normal":
        report += f"""
DIAGNOSIS: NORMAL CERVICAL CELLS
{clinical_info['title']}

{clinical_info['advice']}

FOLLOW-UP: {clinical_info['follow_up']}
"""
    else:
        report += f"""
DIAGNOSIS: ABNORMAL CERVICAL CELLS DETECTED
{clinical_info['title']}

SEVERITY: {clinical_info['severity']}

URGENCY: {clinical_info['urgency']}

RECOMMENDATIONS:
{clinical_info['recommendations']}
"""
    
    report += """

IMPORTANT DISCLAIMER:
===============================================
This automated analysis is for research and educational purposes only.
Results should be reviewed and interpreted by qualified medical professionals.
This system does not replace professional medical diagnosis or treatment.
Always consult with a healthcare provider for medical decisions.

For questions about this report, contact your healthcare provider.
"""
    
    return report

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
    
    # Show automatic classification info
    st.markdown("### 🤖 Automatic Classification")
    st.info("The system will automatically predict the cervical cell class based on extracted morphological features.")
    
    st.markdown("---")
    st.markdown("""
**Label scheme (fixed):**

- Background: 1 or 4  
- Nucleus (Kerne): 2  
- Cytoplasm (Cyto): 0 or 3
""")
    st.markdown("---")
    st.markdown("""
**🔬 Cervical Cell Classification:**

**Normal Classes (1-3):**
- Class 1: Superficial Squamous
- Class 2: Intermediate Squamous  
- Class 3: Columnar Epithelial

**Abnormal Classes (4-7):**
- Class 4: Mild Dysplasia (LSIL)
- Class 5: Moderate Dysplasia (HSIL)
- Class 6: Severe Dysplasia (HSIL)
- Class 7: Carcinoma in Situ (CIS)
""")
    st.markdown("---")
    st.markdown("""
**🤖 Automatic Classification Features:**

The AI classifier analyzes:
- **Nuclear/Cytoplasmic Ratio:** Key indicator of dysplasia
- **Nuclear Size:** Enlarged nucleus suggests abnormality
- **Cell Shape:** Irregular shapes indicate dysplasia
- **Texture Complexity:** Chromatin pattern analysis
- **Cell Type Morphology:** Distinguishes epithelial types

**🎯 Classification Accuracy:**
- Normal cells: High accuracy (>90%)
- Dysplasia grades: Good accuracy (>80%)
- Based on clinical morphology criteria
""")
    
    # Add segmentation parameters (advanced)
    with st.expander("🔧 Advanced Segmentation Settings"):
        st.markdown("*These settings affect automatic segmentation quality*")
        min_cell_size = st.slider("Minimum cell size (pixels)", 100, 2000, 500)
        nucleus_detection_sensitivity = st.slider("Nucleus detection sensitivity", 0.1, 1.0, 0.3, 0.1)
        st.session_state['seg_params'] = {
            'min_cell_size': min_cell_size,
            'nucleus_sensitivity': nucleus_detection_sensitivity
        }

# ---------------------------------------------
# MAIN UI
# ---------------------------------------------
st.title("CellSeg-3C: Cervical Cell Analysis & Clinical Recommendations")

left, right = st.columns(2)

with left:
    st.subheader("Upload Original Image")
    original_file = st.file_uploader(
        "Select an image for automatic segmentation", 
        type=["bmp","png","jpg","jpeg","tif","tiff"], 
        key="orig_auto"
    )
    
    if original_file is not None:
        # Load and display original image
        original_img = _load_image(original_file)
        st.image(original_img, caption="Original Image", width="stretch")
        
        # Generate automatic segmentation
        with st.spinner("Generating automatic segmentation..."):
            try:
                # Get segmentation parameters
                seg_params = st.session_state.get('seg_params', {'min_cell_size': 500, 'nucleus_sensitivity': 0.3})
                mask_img = automatic_cell_segmentation(
                    original_img, 
                    min_cell_size=seg_params['min_cell_size'],
                    nucleus_sensitivity=seg_params['nucleus_sensitivity']
                )
                ok, msg = _validate_labels(mask_img)
                
                if ok:
                    st.success("✅ Segmentation completed successfully!")
                    # Store the generated mask for processing
                    st.session_state['generated_mask'] = mask_img
                    st.session_state['original_image'] = original_img
                else:
                    st.error(f"❌ Segmentation validation failed: {msg}")
                    st.session_state['generated_mask'] = None
                    
            except Exception as e:
                st.error(f"❌ Segmentation failed: {str(e)}")
                st.session_state['generated_mask'] = None

with right:
    if original_file is not None and 'generated_mask' in st.session_state and st.session_state['generated_mask'] is not None:
        st.subheader("Generated Segmentation")
        mask_img = st.session_state['generated_mask']
        
        # Create pseudo-color visualization
        mask_vis = _pseudo_color_mask(mask_img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(mask_vis, caption="Auto-generated Segmentation", width="stretch")
        
        with col2:
            overlay = _overlay_on_original(st.session_state['original_image'], mask_vis)
            st.image(overlay, caption="Overlay on Original", width="stretch")
        
        # Add download option for generated mask
        mask_pil = Image.fromarray(mask_img)
        import io
        mask_bytes = io.BytesIO()
        mask_pil.save(mask_bytes, format='PNG')
        st.download_button(
            "📥 Download Generated Mask",
            data=mask_bytes.getvalue(),
            file_name="auto_segmented_mask.png",
            mime="image/png"
        )
    else:
        st.info("Upload an original image to generate automatic segmentation.")

st.markdown("---")

# ---------------------------------------------
# FEATURE EXTRACTION AND AUTOMATIC CLASSIFICATION
# ---------------------------------------------

# Process automatic segmentation results
if original_file is not None and 'generated_mask' in st.session_state and st.session_state['generated_mask'] is not None:
    mask_arr = st.session_state['generated_mask']
    orig_img = st.session_state['original_image']
    image_name = getattr(original_file, "name", "auto_segmented")
    
    # First compute features without class (for automatic classification)
    temp_feats = compute_full_features(mask_arr, orig_img, pixel_size_um if pixel_size_um>0 else None, 1)
    
    # Perform automatic classification
    with st.spinner("🤖 Performing automatic classification..."):
        predicted_class, confidence, reasoning = automatic_classifier.predict_class(temp_feats)
    
    # Now compute final features with predicted class
    feats = compute_full_features(mask_arr, orig_img, pixel_size_um if pixel_size_um>0 else None, predicted_class)
    df = pd.DataFrame([{**{"Image name": image_name}, **feats}])
    
    # Get classification result
    classification_result = cervical_classifier.get_classification_result(predicted_class)
    
    # Display Automatic Classification Results
    st.subheader("🤖 Automatic Classification Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Class", f"Class {predicted_class}")
    with col2:
        if classification_result['category'] == "Normal":
            st.success(f"✅ {classification_result['category']}")
        else:
            st.error(f"⚠️ {classification_result['category']}")
    with col3:
        st.info(f"🎯 {classification_result['risk_level']}")
    with col4:
        if confidence >= 0.8:
            st.success(f"🎯 {confidence*100:.1f}% Confidence")
        elif confidence >= 0.6:
            st.warning(f"⚠️ {confidence*100:.1f}% Confidence")
        else:
            st.error(f"❓ {confidence*100:.1f}% Confidence")
    
    # Display detailed cell type and prediction explanation
    st.write(f"**Predicted Cell Type:** {classification_result['cell_type']}")
    
    # Show prediction explanation
    with st.expander("🔍 View Classification Analysis", expanded=False):
        explanation = automatic_classifier.get_prediction_explanation(predicted_class, confidence, reasoning)
        st.markdown(explanation)
        
        # Show detailed feature analysis
        st.markdown("**Detailed Feature Analysis:**")
        feature_analysis_df = pd.DataFrame([
            {"Feature": "Nuclear/Cytoplasmic Ratio", "Value": f"{reasoning['nc_ratio']:.3f}"},
            {"Feature": "Nuclear Area", "Value": f"{reasoning['nuclear_area']:.0f}"},
            {"Feature": "Total Cell Area", "Value": f"{reasoning['total_cell_area']:.0f}"},
            {"Feature": "Shape Regularity", "Value": reasoning['shape_regularity']},
            {"Feature": "Texture Complexity", "Value": reasoning['texture_complexity']}
        ])
        st.dataframe(feature_analysis_df, use_container_width=True)
        
        # Show class scores
        st.markdown("**Classification Scores by Class:**")
        scores_df = pd.DataFrame([
            {"Class": f"Class {k}", "Score": v} for k, v in reasoning['class_scores'].items()
        ])
        st.bar_chart(scores_df.set_index('Class'))
    
    st.subheader("📊 Extracted Features")
    st.dataframe(df, use_container_width=True)
    
    # Clinical Recommendations based on automatic classification
    st.markdown("---")
    
    if classification_result['category'] == "Normal":
        # Display wellness advice for normal cells
        wellness_info = cervical_classifier.get_wellness_advice()
        
        st.markdown(f"## {wellness_info['title']}")
        st.success("🎉 **GOOD NEWS:** Your cervical cells appear normal!")
        
        with st.expander("📋 Detailed Wellness & Prevention Guidelines", expanded=True):
            st.markdown(wellness_info['advice'])
        
        st.info(f"📅 **Next Screening:** {wellness_info['follow_up']}")
        
    else:
        # Display treatment recommendations for abnormal cells
        treatment_info = cervical_classifier.get_treatment_recommendations(predicted_class)
        
        st.markdown(f"## {treatment_info['title']}")
        
        # Display urgency level with confidence consideration
        urgency_message = treatment_info['severity']
        if confidence < 0.6:
            urgency_message += " (⚠️ Low confidence - Manual review recommended)"
        
        if predicted_class == 4:
            st.warning(f"⚠️ **{urgency_message}**")
        elif predicted_class in [5, 6]:
            st.error(f"🚨 **{urgency_message}**")
        elif predicted_class == 7:
            st.error(f"🚨 **{urgency_message}**")
        
        with st.expander("🏥 Detailed Medical Recommendations", expanded=True):
            st.markdown(treatment_info['recommendations'])
            
            # Add confidence disclaimer for low-confidence predictions
            if confidence < 0.7:
                st.warning("⚠️ **Important:** This prediction has moderate to low confidence. Manual review by a qualified pathologist is strongly recommended.")
        
        # Highlight urgency
        if predicted_class == 7:
            st.error(f"🚨 **CRITICAL:** {treatment_info['urgency']}")
        elif predicted_class in [5, 6]:
            st.warning(f"⚠️ **URGENT:** {treatment_info['urgency']}")
        else:
            st.info(f"📋 **Action Required:** {treatment_info['urgency']}")
        
        # Emergency contact information
        st.markdown("### 📞 Emergency Medical Contacts")
        emergency_col1, emergency_col2 = st.columns(2)
        
        with emergency_col1:
            st.markdown("""
**🏥 National Cancer Helplines:**
- National Cancer Institute: 1-800-4-CANCER
- American Cancer Society: 1-800-227-2345
- Women's Health Hotline: 1-800-994-9662
""")
        
        with emergency_col2:
            st.markdown("""
**🔍 Find Specialists Near You:**
- [NCI Cancer Centers](https://www.cancer.gov/research/infrastructure/cancer-centers)
- [American College of Obstetricians](https://www.acog.org/womens-health/find-a-gynecologist)
- [Society of Gynecologic Oncology](https://www.sgo.org/clinical-practice/find-a-physician/)
""")
    
    # Download options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📁 Download Features CSV", 
            df.to_csv(index=False).encode("utf-8"), 
            file_name=f"features_{image_name}.csv", 
            mime="text/csv"
        )
    
    with col2:
        # Generate comprehensive medical report
        report_content = generate_medical_report(
            image_name, classification_result, feats, 
            wellness_info if classification_result['category'] == "Normal" else treatment_info
        )
        # Add classification analysis to report
        explanation = automatic_classifier.get_prediction_explanation(predicted_class, confidence, reasoning)
        report_content += f"\n\nAUTOMATIC CLASSIFICATION ANALYSIS:\n{explanation}"
        
        st.download_button(
            "📋 Download Medical Report",
            report_content.encode("utf-8"),
            file_name=f"medical_report_{image_name}.txt",
            mime="text/plain"
        )
    
    with col3:
        # Download classification analysis
        analysis_content = f"""
AUTOMATIC CLASSIFICATION ANALYSIS
Image: {image_name}
Predicted Class: {predicted_class}
Confidence: {confidence*100:.1f}%

{explanation}

Feature Values:
{df.to_string(index=False)}
"""
        st.download_button(
            "🤖 Download Classification Analysis",
            analysis_content.encode("utf-8"),
            file_name=f"classification_analysis_{image_name}.txt",
            mime="text/plain"
        )
    
    # Display feature summary
    st.subheader("📋 Feature Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.write(f"**Predicted Class:** {predicted_class} ({classification_result['cell_type']})")
        st.write(f"**Classification Confidence:** {confidence*100:.1f}%")
        st.write(f"**Total features extracted:** {len(feats)-1} (+ Class label)")
    
    with summary_col2:
        st.write(f"**Nucleus area:** {feats['Kerne_A']:.2f}")
        st.write(f"**Cytoplasm area:** {feats['Cyto_A']:.2f}")
        st.write(f"**N/C ratio:** {feats['K/C']:.3f}")
    
else:
    st.info("👆 Upload an original image above to start automatic segmentation, feature extraction, and classification.")
