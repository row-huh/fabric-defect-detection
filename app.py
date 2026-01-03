import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="Car Fabric Defect Detection",
    layout="wide"
)

def detect_car_fabric_defects(image):
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        vis_image = img_array.copy()
    else:
        gray = img_array
        vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    texture_diff = cv2.absdiff(gray, blurred)
    mean_texture = np.mean(texture_diff)
    std_texture = np.std(texture_diff)
    
    threshold_value = max(30, mean_texture + 3.5 * std_texture)
    
    _, anomalies = cv2.threshold(texture_diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    dents = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 101, 15
    )

    
    total_defects = cv2.bitwise_or(anomalies, dents)
    
    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    total_defects = cv2.morphologyEx(total_defects, cv2.MORPH_OPEN, kernel)
    total_defects = cv2.dilate(total_defects, kernel, iterations=2)

    
    defect_count = 0
    contours, _ = cv2.findContours(total_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
    
        if 100 < area < (gray.shape[0] * gray.shape[1] * 0.5):
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
            defect_count += 1

    is_flawed = defect_count > 0

    return {
        'is_flawed': is_flawed,
        'defect_count': defect_count,
        'visualization': vis_image,
        'blurred': blurred,
        'texture_diff': texture_diff,
        'defect_mask': total_defects
    }

st.title("Car Fabric Defect Detection")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


st.sidebar.header("Detection Settings")
show_debug = st.sidebar.checkbox("Show Debug Layers", value=False)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.success("Image uploaded successfully!")
    
    with st.spinner("Analyzing car fabric..."):
        results = detect_car_fabric_defects(image)
    
    st.markdown("---")
    
    if results['is_flawed']:
        st.error(f"**DEFECTS FOUND**: {results['defect_count']} issues detected")
    else:
        st.success("**PASS**: Fabric appears clean")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Defect Visualization")
        st.image(results['visualization'], use_container_width=True, caption="Detected Defects (Red Boxes)")
        
    if show_debug:
        st.markdown("---")
        st.subheader("Debug Layers (Statistical Analysis)")
        
        d1, d2, d3 = st.columns(3)
        
        with d1:
            st.image(results['blurred'], caption="Shape (Texture Removed)", use_container_width=True)
        with d2:
            st.image(results['texture_diff'], caption="Texture Energy", use_container_width=True)
        with d3:
            st.image(results['defect_mask'], caption="Statistical Outliers", use_container_width=True)

else:
    st.info("Please upload a car door fabric image to begin")
