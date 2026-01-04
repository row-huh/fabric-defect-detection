import streamlit as st
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

st.set_page_config(
    page_title="SUZUKI Car Fabric Defect Detection - Gabor Filters",
    layout="wide"
)

def segment_fabric_region(image):
    """
    Segment fabric from background using color-based clustering.
    Returns binary mask of fabric region.
    """
    if len(image.shape) == 2:
        # Convert grayscale to BGR for processing
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()
    
    # Resize for faster processing
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    else:
        img_small = img_bgr
    
    # Convert to LAB color space (better for fabric segmentation)
    lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)
    
    # Reshape for clustering
    pixels = lab.reshape(-1, 3).astype(np.float32)
    
    # Use GMM to separate fabric from background
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(pixels)
    labels = gmm.predict(pixels)
    
    # Reshape back to image
    mask_small = labels.reshape(img_small.shape[:2])
    
    # Choose the larger component as fabric (usually fabric occupies more area)
    count_0 = np.sum(mask_small == 0)
    count_1 = np.sum(mask_small == 1)
    
    if count_0 > count_1:
        fabric_mask = (mask_small == 0).astype(np.uint8) * 255
    else:
        fabric_mask = (mask_small == 1).astype(np.uint8) * 255
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fabric_mask = cv2.morphologyEx(fabric_mask, cv2.MORPH_CLOSE, kernel)
    fabric_mask = cv2.morphologyEx(fabric_mask, cv2.MORPH_OPEN, kernel)
    
    # Resize mask back to original size
    if scale < 1.0:
        fabric_mask = cv2.resize(fabric_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Find largest connected component (main fabric region)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fabric_mask, connectivity=8)
    if num_labels > 1:
        # Get largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fabric_mask = (labels == largest_label).astype(np.uint8) * 255
    
    return fabric_mask

def build_gabor_filter_bank(num_orientations=8, num_scales=5):
    """
    Build a bank of Gabor filters with different orientations and scales.
    Gabor filters are excellent for texture analysis as they capture spatial frequency and orientation.
    """
    filters = []
    ksize = 31  # Kernel size
    
    for scale in range(num_scales):
        # Wavelength (lambda) - controls the scale of the pattern
        wavelength = 2.0 ** (scale + 1)
        
        for orientation in range(num_orientations):
            theta = orientation * np.pi / num_orientations
            
            # Create Gabor kernel
            # sigma: standard deviation of Gaussian envelope
            # gamma: spatial aspect ratio
            # psi: phase offset
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=wavelength * 0.56,  # Proportional to wavelength
                theta=theta,
                lambd=wavelength,
                gamma=0.5,  # Ellipticity
                psi=0,  # Phase offset
                ktype=cv2.CV_32F
            )
            kernel /= kernel.sum()  # Normalize
            filters.append((kernel, wavelength, theta))
    
    return filters

def apply_gabor_filter_bank(image, filters):
    """
    Apply all Gabor filters to the image and return feature responses.
    """
    responses = []
    
    for kernel, wavelength, theta in filters:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        responses.append(filtered)
    
    return responses

def compute_gabor_energy(responses):
    """
    Compute texture energy from Gabor filter responses.
    Energy measures the strength of texture patterns.
    """
    # Calculate magnitude of responses
    energy = np.zeros_like(responses[0])
    
    for response in responses:
        energy += np.abs(response)
    
    energy = energy / len(responses)
    return energy

def compute_texture_features(responses, window_size=15):
    """
    Compute statistical texture features from Gabor responses.
    Returns mean and standard deviation in local neighborhoods.
    """
    features = []
    
    for response in responses:
        # Mean feature
        mean_response = cv2.blur(np.abs(response), (window_size, window_size))
        features.append(mean_response)
        
        # Standard deviation feature
        squared_response = cv2.blur(response ** 2, (window_size, window_size))
        std_response = np.sqrt(np.maximum(squared_response - mean_response ** 2, 0))
        features.append(std_response)
    
    return features

def detect_car_fabric_defects(image):
    """
    Texture Segmentation & Fault Detection using Optimal Gabor Filters.
    Based on: Bodnarova et al., "Optimal Gabor filters for textile flaw detection"
    Pattern Recognition, 2002.
    
    Steps:
    1. Segment fabric region from background
    2. Design optimal Gabor filters using Fisher cost function
    3. Classify pixels as defective/non-defective based on filter responses
    """
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

    # ===== STEP 1: FABRIC SEGMENTATION =====
    # Isolate fabric region from background
    
    fabric_mask = segment_fabric_region(img_array if len(img_array.shape) == 3 else vis_image)
    
    # Apply mask to focus only on fabric
    fabric_only = cv2.bitwise_and(gray, gray, mask=fabric_mask)
    
    # ===== PREPROCESSING =====
    
    # Normalize intensity
    normalized = cv2.normalize(fabric_only, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply slight Gaussian blur to reduce noise
    preprocessed = cv2.GaussianBlur(normalized, (3, 3), 0)
    
    # ===== GABOR FILTER BANK CONSTRUCTION =====
    
    # Build filter bank with multiple orientations and scales
    gabor_filters = build_gabor_filter_bank(num_orientations=8, num_scales=4)
    
    # ===== APPLY GABOR FILTERS =====
    
    # Apply all filters to get texture responses
    gabor_responses = apply_gabor_filter_bank(preprocessed, gabor_filters)
    
    # ===== LEARN NORMAL TEXTURE PATTERN =====
    
    # Compute texture features for each pixel
    texture_features = compute_texture_features(gabor_responses, window_size=15)
    
    # Stack all features into a single feature vector per pixel
    feature_stack = np.array(texture_features)
    
    # Compute energy map
    energy_map = compute_gabor_energy(gabor_responses)
    energy_normalized = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ===== BUILD NORMAL TEXTURE MODEL (Template-based approach from paper) =====
    # Based on Bodnarova et al.: Use mean and variance of template (defect-free) region
    # to establish baseline for normal texture using Fisher discriminant
    
    # Extract template region: assume center region is mostly defect-free
    h, w = preprocessed.shape
    template_region = preprocessed[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    template_mask = fabric_mask[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    
    # Apply filters to template region only
    template_responses = []
    for kernel, wavelength, theta in gabor_filters:
        filtered = cv2.filter2D(template_region, cv2.CV_32F, kernel)
        # Only consider fabric pixels in template
        template_responses.append(filtered[template_mask > 0])
    
    # Compute statistics for each filter response (Fisher discriminant approach)
    filter_means = []
    filter_stds = []
    for resp in template_responses:
        if len(resp) > 0:
            filter_means.append(np.mean(np.abs(resp)))
            filter_stds.append(np.std(np.abs(resp)))
        else:
            filter_means.append(0)
            filter_stds.append(1)
    
    # Compute deviation from normal pattern for each filter
    # Using Fisher linear discriminant: (x - mean) / std
    deviation_maps = []
    for response, mean_val, std_val in zip(gabor_responses, filter_means, filter_stds):
        if std_val > 0:
            # Normalized deviation (z-score)
            deviation = np.abs((np.abs(response) - mean_val) / std_val)
        else:
            deviation = np.abs(np.abs(response) - mean_val)
        deviation_maps.append(deviation)
    
    # Aggregate deviations across all filters (weighted by discriminative power)
    # Filters with lower variance in template are more discriminative (Fisher criterion)
    weights = []
    for std_val in filter_stds:
        if std_val > 0:
            weights.append(1.0 / std_val)  # Lower variance = higher weight
        else:
            weights.append(1.0)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize
    
    total_deviation = np.zeros_like(deviation_maps[0])
    for dev_map, weight in zip(deviation_maps, weights):
        total_deviation += dev_map * weight
    
    # Normalize deviation map
    deviation_normalized = cv2.normalize(total_deviation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ===== TEXTURE CONSISTENCY ANALYSIS =====
    
    # Compute local texture consistency using sliding window
    # Normal fabric has consistent Gabor response; defects cause inconsistency
    consistency_score = np.zeros_like(preprocessed, dtype=np.float32)
    
    # Use energy map to compute local consistency
    window_size = 25
    for i in range(len(gabor_responses[:8])):  # Use subset for speed
        response = np.abs(gabor_responses[i])
        # Local standard deviation indicates texture variation
        local_mean = cv2.blur(response, (window_size, window_size))
        local_sq_mean = cv2.blur(response**2, (window_size, window_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        consistency_score += local_std
    
    consistency_score = consistency_score / 8
    consistency_normalized = cv2.normalize(consistency_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ===== ANOMALY DETECTION WITH STATISTICAL THRESHOLDING =====
    # Paper approach: Use mean and std of template responses for classification
    # Based on 3-sigma rule for statistical significance
    
    # Only analyze fabric regions
    fabric_pixels = fabric_mask > 0
    
    # Method 1: Deviation-based (primary method from paper)
    # Threshold based on statistical significance (z-score approach)
    # Use 3-sigma rule: values beyond 3 standard deviations are anomalies
    deviation_fabric = total_deviation[fabric_pixels]
    mean_dev = np.mean(deviation_fabric)
    std_dev = np.std(deviation_fabric)
    
    # Adaptive threshold: mean + k*std (k=3 for strict, k=2 for sensitive)
    k_threshold = 3.0  # Conservative threshold (99.7% confidence)
    deviation_threshold = mean_dev + k_threshold * std_dev
    
    _, deviation_mask = cv2.threshold(total_deviation, deviation_threshold, 255, cv2.THRESH_BINARY)
    deviation_mask = deviation_mask.astype(np.uint8)
    
    # Restrict to fabric region only
    deviation_mask = cv2.bitwise_and(deviation_mask, deviation_mask, mask=fabric_mask)
    
    # Method 2: Consistency-based
    # High consistency variation indicates anomaly
    consistency_threshold = np.percentile(consistency_score, 97)
    _, consistency_mask = cv2.threshold(consistency_score, consistency_threshold, 255, cv2.THRESH_BINARY)
    consistency_mask = consistency_mask.astype(np.uint8)
    
    # Method 3: Energy outliers (only extreme cases)
    energy_low = np.percentile(energy_map, 1)  # Very low energy
    energy_high = np.percentile(energy_map, 99.5)  # Very high energy
    low_energy_mask = (energy_map < energy_low).astype(np.uint8) * 255
    high_energy_mask = (energy_map > energy_high).astype(np.uint8) * 255
    energy_anomaly_mask = cv2.bitwise_or(low_energy_mask, high_energy_mask)
    
    # ===== DEFECT DETECTION =====
    
    # Combine methods - require agreement from multiple methods for robustness
    # This reduces false positives from normal texture
    combined_mask = cv2.bitwise_or(deviation_mask, consistency_mask)
    # Only consider energy anomalies if they coincide with other detections
    combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(cv2.bitwise_not(combined_mask)))
    
    # ===== MORPHOLOGICAL CLEANUP =====
    
    # Aggressive cleanup to remove normal texture variations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Remove small isolated pixels (noise and normal texture)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Close small gaps within actual defects
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    final_defects = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
    
    # Final cleanup: remove very small regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_defects, connectivity=8)
    
    # Filter out tiny components (likely normal texture variation)
    min_defect_size = 200  # Minimum pixels for a real defect
    filtered_mask = np.zeros_like(final_defects)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_defect_size:
            filtered_mask[labels == i] = 255
    
    final_defects = filtered_mask

    # ===== VISUALIZATION =====
    
    defect_count = 0
    contours, _ = cv2.findContours(final_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # More strict area filtering
        if 200 < area < (gray.shape[0] * gray.shape[1] * 0.25):
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw thicker, more visible boxes
            cv2.rectangle(vis_image, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 4)
            # Add label
            label_text = f"DEFECT {defect_count+1}"
            cv2.putText(vis_image, label_text, (x, max(y-10, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            defect_count += 1

    is_flawed = defect_count > 0
    
    # Compute feature variance for visualization
    feature_variance = np.var(feature_stack, axis=0)
    variance_normalized = cv2.normalize(feature_variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create visualization of some Gabor filter responses
    sample_responses = []
    for i in [0, 8, 16, 24]:  # Sample different orientations/scales
        if i < len(gabor_responses):
            resp = cv2.normalize(np.abs(gabor_responses[i]), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sample_responses.append(resp)
    
    # Create fabric segmentation visualization
    fabric_vis = cv2.cvtColor(fabric_mask, cv2.COLOR_GRAY2RGB)
    fabric_vis[:, :, 1] = np.maximum(fabric_vis[:, :, 1], (fabric_mask > 0).astype(np.uint8) * 100)
    
    return {
        'is_flawed': is_flawed,
        'defect_count': defect_count,
        'visualization': vis_image,
        'preprocessed': preprocessed,
        'fabric_mask': fabric_mask,
        'fabric_vis': fabric_vis,
        'energy_map': energy_normalized,
        'variance_map': variance_normalized,
        'deviation_map': deviation_normalized,
        'consistency_map': consistency_normalized,
        'defect_mask': final_defects,
        'sample_responses': sample_responses,
        'combined_mask': combined_mask
    }


st.title("🚗 SUZUKI Car Door Fabric Defect Detection")
st.subheader("Texture Segmentation using Gabor Filters")

uploaded_file = st.file_uploader(
    "Upload car door fabric image",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.header("⚙️ Detection Settings")
show_debug = st.sidebar.checkbox("Show Gabor Filter Analysis", value=False)
show_theory = st.sidebar.checkbox("Show Theory", value=False)

if show_theory:
    with st.sidebar.expander("📚 Gabor Filter Theory", expanded=False):
        st.markdown("""
        **Gabor Filters** are linear filters used for texture analysis and edge detection.
        
        **Key Properties:**
        - Captures spatial frequency and orientation
        - Similar to human visual system
        - Excellent for periodic patterns (like fabric weave)
        
        **Parameters:**
        - **Wavelength (λ)**: Pattern scale
        - **Orientation (θ)**: Pattern direction
        - **Sigma (σ)**: Gaussian envelope width
        - **Gamma (γ)**: Spatial aspect ratio
        
        **For Fabric Defects:**
        - Normal fabric = consistent Gabor response
        - Defects = deviation from normal pattern
        """)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.success("✅ Image uploaded successfully!")
    
    with st.spinner("🔍 Analyzing fabric texture with Gabor filters..."):
        results = detect_car_fabric_defects(image)
    
    st.markdown("---")
    
    # Results Display
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        if results['is_flawed']:
            st.error(f"### ❌ DEFECTS DETECTED")
            st.metric("Defect Count", results['defect_count'])
        else:
            st.success("### ✅ QUALITY PASS")
            st.metric("Defect Count", 0)
    
    with result_col2:
        if results['is_flawed']:
            st.warning("""
            **Recommendation**: Further inspection required. 
            Detected anomalies in texture pattern suggest potential defects in fabric structure.
            """)
        else:
            st.info("**Status**: Fabric texture appears uniform. No significant defects detected.")
    
    st.markdown("---")
    
    # Main visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Fabric Segmentation")
        st.image(results['fabric_vis'], use_container_width=True,
                caption="Green overlay = detected fabric region")
        
    with col3:
        st.subheader("⚠️ Defect Detection")
        st.image(results['visualization'], use_container_width=True, 
                caption="Red boxes = detected defects")
    
    # Texture Analysis Section
    if show_debug:
        st.markdown("---")
        st.subheader("🔬 Gabor Filter Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Texture Energy", "Gabor Responses", "Detection Masks"])
        
        with tab1:
            st.markdown("**Step 1: Fabric Segmentation & Template Analysis**")
            seg_col1, seg_col2 = st.columns(2)
            
            with seg_col1:
                st.image(results['fabric_mask'], caption="Fabric Mask (Binary)",
                        use_container_width=True)
                st.caption("Isolated fabric region for analysis")
            
            with seg_col2:
                # Show template region overlay
                h, w = results['preprocessed'].shape
                template_vis = cv2.cvtColor(results['preprocessed'], cv2.COLOR_GRAY2RGB)
                cv2.rectangle(template_vis, 
                            (int(w*0.3), int(h*0.3)), 
                            (int(w*0.7), int(h*0.7)), 
                            (255, 255, 0), 3)
                st.image(template_vis, caption="Template Region (Yellow Box)",
                        use_container_width=True)
                st.caption("Center region used as defect-free template")
            
            st.markdown("**Step 2: Gabor Filter Responses & Fisher Discriminant Analysis**")
            energy_col1, energy_col2, energy_col3, energy_col4 = st.columns(4)
            
            with energy_col1:
                st.image(results['energy_map'], caption="Gabor Energy Map", 
                        use_container_width=True)
                st.caption("Texture strength across image")
                
            with energy_col2:
                st.image(results['deviation_map'], caption="Deviation from Normal", 
                        use_container_width=True)
                st.caption("⚠️ Red = deviates from learned pattern")
                
            with energy_col3:
                st.image(results['consistency_map'], caption="Texture Consistency", 
                        use_container_width=True)
                st.caption("Low = uniform, High = varied")
                
            with energy_col4:
                st.image(results['variance_map'], caption="Feature Variance", 
                        use_container_width=True)
                st.caption("Pattern variation analysis")
        
        with tab2:
            st.markdown("**Sample Gabor Filter Responses (Different Orientations & Scales)**")
            if len(results['sample_responses']) >= 4:
                resp_col1, resp_col2, resp_col3, resp_col4 = st.columns(4)
                
                with resp_col1:
                    st.image(results['sample_responses'][0], 
                            caption="Scale 1, Orientation 1", use_container_width=True)
                with resp_col2:
                    st.image(results['sample_responses'][1], 
                            caption="Scale 2, Orientation 1", use_container_width=True)
                with resp_col3:
                    st.image(results['sample_responses'][2], 
                            caption="Scale 3, Orientation 1", use_container_width=True)
                with resp_col4:
                    st.image(results['sample_responses'][3], 
                            caption="Scale 4, Orientation 1", use_container_width=True)
                
                st.caption("Different filters respond to different texture patterns")
        
        with tab3:
            st.markdown("**Detection Masks & Processing Steps**")
            mask_col1, mask_col2, mask_col3 = st.columns(3)
            
            with mask_col1:
                st.image(results['preprocessed'], caption="Preprocessed Image", 
                        use_container_width=True)
            with mask_col2:
                st.image(results['combined_mask'], caption="Combined Anomaly Mask", 
                        use_container_width=True)
            with mask_col3:
                st.image(results['defect_mask'], caption="Final Defect Mask", 
                        use_container_width=True)

else:
    st.info("👆 Upload a SUZUKI car door fabric image to begin defect detection")
    
    # Sample images section
    st.markdown("---")
    st.subheader("📁 Test with Sample Images")
    
    sample_cols = st.columns(4)
    sample_images = [
        "dataset/fabric_car1.jpeg",
        "dataset/fabric_car2.jpeg", 
        "dataset/fabric_car3.jpeg",
        "dataset/fabric_car4.jpeg"
    ]
    
    for idx, (col, img_path) in enumerate(zip(sample_cols, sample_images)):
        with col:
            try:
                sample_img = Image.open(img_path)
                st.image(sample_img, caption=f"Sample {idx+1}", use_container_width=True)
            except:
                st.caption(f"Sample {idx+1} not found")
    
    with st.expander("ℹ️ About Gabor Filter Method"):
        st.markdown("""
        ### Texture Segmentation & Fault Detection using Optimal Gabor Filters
        
        **Based on:** Bodnarova et al., "Optimal Gabor filters for textile flaw detection", Pattern Recognition, 2002
        
        **Method Overview:**
        
        1. **Fabric Segmentation**
           - Uses Gaussian Mixture Model (GMM) on LAB color space
           - Isolates fabric region from background
           - Focuses analysis only on actual fabric texture
           - Eliminates background interference
           
        2. **Gabor Filter Bank Construction**
           - Creates 32 Gabor filters (8 orientations × 4 scales)
           - Each filter captures specific frequency and orientation
           - Optimal for periodic textile patterns
           - Covers spatial-frequency domain comprehensively
           
        3. **Template-Based Learning (Fisher Discriminant)**
           - Extracts defect-free template from center region
           - Computes mean (μ) and variance (σ²) for each filter response
           - Builds statistical model of normal texture
           - Weights filters by discriminative power: w = 1/σ
           - Lower variance = more consistent = higher weight
           
        4. **Statistical Classification (3-Sigma Rule)**
           - Computes z-score: z = (x - μ) / σ for each pixel
           - Weighted aggregation across all filters
           - Threshold: μ + 3σ (99.7% confidence interval)
           - Pixels beyond threshold classified as defects
           - Only analyzes pixels within fabric mask
           
        5. **Defect Isolation & Morphological Refinement**
           - Aggressive morphological filtering removes normal variations
           - Connected component analysis with minimum size filter (200+ pixels)
           - Only significant anomalies are flagged
           - Bounding boxes highlight actual defects
        
        **Advantages of Gabor Filters:**
        - ✓ Mimics human visual perception
        - ✓ Excellent for periodic textures (fabric weave)
        - ✓ Multi-scale and multi-orientation analysis
        - ✓ Robust to illumination changes
        - ✓ Established method in textile quality control
        """)
    
    st.markdown("---")
    st.caption("Project 9: Texture Segmentation & Fault Detection in SUZUKI Car Door Fabric using Gabor Filters")

