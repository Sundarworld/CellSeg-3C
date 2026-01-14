"""
CellSeg-3C - Ultra Stable Version
Cervical Cell Analysis with Error Handling
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="CellSeg-3C Stable", 
    page_icon="🔬",
    layout="wide"
)

def safe_import_cv2():
    """Safely import cv2 with fallback"""
    try:
        import cv2
        return cv2, True
    except ImportError:
        return None, False

def safe_import_skimage():
    """Safely import skimage with fallback"""
    try:
        from skimage import measure, morphology, filters
        return (measure, morphology, filters), True
    except ImportError:
        return (None, None, None), False

# Try imports
cv2, has_cv2 = safe_import_cv2()
(measure, morphology, filters), has_skimage = safe_import_skimage()

def load_image_safely(uploaded_file):
    """Safely load uploaded image"""
    try:
        if uploaded_file is None:
            return None, "No file uploaded"
        
        # Load image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array, "Success"
    
    except Exception as e:
        return None, f"Error loading image: {str(e)}"

def simple_segmentation(image):
    """Simple segmentation without complex dependencies"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # Simple thresholding
        threshold = np.mean(gray)
        binary = (gray < threshold).astype(np.uint8)
        
        # Create segmentation mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[binary == 1] = 2  # Nucleus (dark regions)
        mask[binary == 0] = 1  # Background (light regions)
        
        return mask, True, "Simple segmentation completed"
    
    except Exception as e:
        return np.zeros_like(image[:,:,0] if len(image.shape) == 3 else image, dtype=np.uint8), False, f"Segmentation failed: {str(e)}"

def advanced_segmentation(image):
    """Advanced segmentation using cv2 if available"""
    if not has_cv2:
        return simple_segmentation(image)
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Create mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[binary > 0] = 2  # Nucleus
        mask[binary == 0] = 1  # Background
        
        return mask, True, "Advanced segmentation completed"
    
    except Exception as e:
        return simple_segmentation(image)

def extract_basic_features(mask, original_image):
    """Extract basic morphological features"""
    try:
        features = {}
        
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            gray = np.mean(original_image, axis=2)
        else:
            gray = original_image.copy()
        
        # Get regions
        nucleus_mask = (mask == 2)
        background_mask = (mask == 1)
        
        # Basic measurements
        nucleus_area = np.sum(nucleus_mask)
        total_area = mask.size
        background_area = np.sum(background_mask)
        cytoplasm_area = total_area - nucleus_area - background_area
        
        # Features
        features["Nucleus_Area"] = int(nucleus_area)
        features["Cytoplasm_Area"] = int(cytoplasm_area)
        features["Total_Area"] = int(total_area)
        features["NC_Ratio"] = float(nucleus_area / max(cytoplasm_area, 1))
        
        # Intensity features
        if nucleus_area > 0:
            nucleus_intensities = gray[nucleus_mask]
            features["Nucleus_Mean_Intensity"] = float(np.mean(nucleus_intensities))
            features["Nucleus_Std_Intensity"] = float(np.std(nucleus_intensities))
        else:
            features["Nucleus_Mean_Intensity"] = 0.0
            features["Nucleus_Std_Intensity"] = 0.0
        
        if cytoplasm_area > 0:
            cytoplasm_mask_bool = (mask == 0) | ((mask != 1) & (mask != 2))
            if np.any(cytoplasm_mask_bool):
                cytoplasm_intensities = gray[cytoplasm_mask_bool]
                features["Cytoplasm_Mean_Intensity"] = float(np.mean(cytoplasm_intensities))
                features["Cytoplasm_Std_Intensity"] = float(np.std(cytoplasm_intensities))
            else:
                features["Cytoplasm_Mean_Intensity"] = 0.0
                features["Cytoplasm_Std_Intensity"] = 0.0
        else:
            features["Cytoplasm_Mean_Intensity"] = 0.0
            features["Cytoplasm_Std_Intensity"] = 0.0
        
        # Additional basic features
        features["Nucleus_Coverage_Percent"] = float(nucleus_area / total_area * 100)
        features["Cytoplasm_Coverage_Percent"] = float(cytoplasm_area / total_area * 100)
        
        return features, True, "Feature extraction successful"
    
    except Exception as e:
        return {}, False, f"Feature extraction failed: {str(e)}"

def classify_cell(features):
    """Simple rule-based classification"""
    try:
        nc_ratio = features.get("NC_Ratio", 0)
        nucleus_coverage = features.get("Nucleus_Coverage_Percent", 0)
        
        # Simple classification rules
        if nc_ratio < 0.2:
            predicted_class = 1
            cell_type = "Normal Superficial"
            confidence = 0.85
        elif nc_ratio < 0.4:
            predicted_class = 2
            cell_type = "Normal Intermediate"
            confidence = 0.80
        elif nc_ratio < 0.6:
            predicted_class = 3
            cell_type = "Normal Columnar"
            confidence = 0.75
        elif nc_ratio < 0.8:
            predicted_class = 4
            cell_type = "Mild Dysplasia"
            confidence = 0.70
        else:
            predicted_class = 7
            cell_type = "Severe Abnormality"
            confidence = 0.65
        
        return predicted_class, cell_type, confidence
    
    except Exception as e:
        return 1, "Unknown", 0.5

def create_colored_mask(mask):
    """Create colored visualization of mask"""
    try:
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # Colors: Background=black, Nucleus=red, Cytoplasm=blue
        colored_mask[mask == 1] = [0, 0, 0]      # Background - black
        colored_mask[mask == 2] = [255, 0, 0]    # Nucleus - red
        colored_mask[mask == 0] = [0, 0, 255]    # Cytoplasm - blue
        
        return colored_mask
    except Exception as e:
        return np.zeros((100, 100, 3), dtype=np.uint8)

def main():
    """Main application"""
    
    # Title and header
    st.title("🔬 CellSeg-3C: Stable Cervical Cell Analysis")
    st.markdown("**Robust Implementation with Error Handling**")
    
    # Check dependencies
    with st.expander("📦 System Information", expanded=False):
        st.write(f"OpenCV Available: {'✅ Yes' if has_cv2 else '❌ No (using fallback)'}")
        st.write(f"Scikit-image Available: {'✅ Yes' if has_skimage else '❌ No (using fallback)'}")
        st.write("Core Libraries: ✅ NumPy, Pandas, PIL")
    
    st.markdown("---")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        use_advanced = st.checkbox("Use Advanced Segmentation", value=has_cv2, disabled=not has_cv2)
        show_debug = st.checkbox("Show Debug Information", value=False)
    
    # File upload
    st.subheader("📤 Upload Cell Image")
    uploaded_file = st.file_uploader(
        "Choose a cervical cell image...", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a PAP smear or cervical cell image"
    )
    
    if uploaded_file is not None:
        # Load image
        with st.spinner("Loading image..."):
            original_image, load_status = load_image_safely(uploaded_file)
        
        if original_image is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(original_image, caption=f"Original Image ({original_image.shape})", use_container_width=True)
                
                if show_debug:
                    st.write(f"Shape: {original_image.shape}")
                    st.write(f"Type: {original_image.dtype}")
                    st.write(f"Range: {original_image.min()}-{original_image.max()}")
            
            # Process button
            if st.button("🚀 Analyze Cell", type="primary"):
                
                # Step 1: Segmentation
                with st.spinner("Performing segmentation..."):
                    if use_advanced and has_cv2:
                        mask, seg_success, seg_message = advanced_segmentation(original_image)
                    else:
                        mask, seg_success, seg_message = simple_segmentation(original_image)
                
                if seg_success:
                    st.success(f"✅ {seg_message}")
                    
                    # Display segmentation
                    with col2:
                        st.subheader("🎯 Segmentation Result")
                        colored_mask = create_colored_mask(mask)
                        st.image(colored_mask, caption="Segmented Mask (Red=Nucleus, Blue=Cytoplasm, Black=Background)", use_container_width=True)
                    
                    # Step 2: Feature extraction
                    with st.spinner("Extracting features..."):
                        features, feat_success, feat_message = extract_basic_features(mask, original_image)
                    
                    if feat_success:
                        st.success(f"✅ {feat_message}")
                        
                        # Step 3: Classification
                        with st.spinner("Classifying cell..."):
                            predicted_class, cell_type, confidence = classify_cell(features)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        # Classification summary
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            st.metric("Cell Class", f"Class {predicted_class}")
                            
                        with result_col2:
                            st.metric("Cell Type", cell_type)
                            
                        with result_col3:
                            if confidence >= 0.8:
                                st.success(f"🎯 {confidence*100:.1f}% Confidence")
                            elif confidence >= 0.6:
                                st.warning(f"⚠️ {confidence*100:.1f}% Confidence")
                            else:
                                st.error(f"❓ {confidence*100:.1f}% Confidence")
                        
                        # Clinical interpretation
                        if predicted_class <= 3:
                            st.success("✅ **Normal Cell** - No immediate concerns")
                        else:
                            st.warning("⚠️ **Abnormal Cell** - May require medical attention")
                        
                        # Feature table
                        st.subheader("📋 Extracted Features")
                        features_df = pd.DataFrame([features])
                        st.dataframe(features_df, use_container_width=True)
                        
                        # Key measurements
                        with st.expander("🔍 Key Measurements", expanded=True):
                            meas_col1, meas_col2 = st.columns(2)
                            
                            with meas_col1:
                                st.write("**Morphological Features:**")
                                st.write(f"• Nuclear Area: {features['Nucleus_Area']:,} pixels")
                                st.write(f"• Cytoplasm Area: {features['Cytoplasm_Area']:,} pixels")
                                st.write(f"• N/C Ratio: {features['NC_Ratio']:.3f}")
                            
                            with meas_col2:
                                st.write("**Intensity Features:**")
                                st.write(f"• Nuclear Mean Intensity: {features['Nucleus_Mean_Intensity']:.1f}")
                                st.write(f"• Cytoplasm Mean Intensity: {features['Cytoplasm_Mean_Intensity']:.1f}")
                                st.write(f"• Nuclear Coverage: {features['Nucleus_Coverage_Percent']:.1f}%")
                        
                        # Download options
                        st.markdown("---")
                        st.subheader("📥 Download Results")
                        
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            # Download segmentation
                            mask_img = Image.fromarray(colored_mask)
                            mask_bytes = io.BytesIO()
                            mask_img.save(mask_bytes, format='PNG')
                            st.download_button(
                                "Download Segmentation",
                                data=mask_bytes.getvalue(),
                                file_name=f"segmentation_{uploaded_file.name}.png",
                                mime="image/png"
                            )
                        
                        with download_col2:
                            # Download features
                            csv_buffer = io.StringIO()
                            features_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                "Download Features CSV", 
                                data=csv_buffer.getvalue(),
                                file_name=f"features_{uploaded_file.name}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        st.error(f"❌ {feat_message}")
                
                else:
                    st.error(f"❌ {seg_message}")
        
        else:
            st.error(f"❌ {load_status}")
    
    else:
        st.info("👆 Upload a cervical cell image to start analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("**CellSeg-3C Stable Version** - Robust Cervical Cell Analysis System")

if __name__ == "__main__":
    main()
