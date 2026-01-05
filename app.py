import os
from glob import glob
import numpy as np
import cv2
import streamlit as st
from skimage.filters import gabor
from skimage.filters.rank import entropy
from skimage.morphology import disk
from PIL import Image


def load_image(path):
	img = cv2.imread(path)
	if img is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_fabric_region(img_rgb, mode="auto"):
	gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	h, w = gray.shape
	# Texture prior via entropy
	ent = entropy(gray, disk(9)).astype(np.float32)
	ent_norm = cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	_, th_ent = cv2.threshold(ent_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	th_ent = cv2.morphologyEx(th_ent, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
	th_ent = cv2.morphologyEx(th_ent, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

	# Color prior via k-means (HSV space)
	hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
	hsv_small = cv2.resize(hsv, (w // 4 if w > 4 else w, h // 4 if h > 4 else h), interpolation=cv2.INTER_AREA)
	Z = hsv_small.reshape((-1, 3)).astype(np.float32)
	K = 3
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
	_, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
	labels = labels.reshape(hsv_small.shape[:2])
	# Upsample label map back to original size
	label_full = cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

	# Choose the cluster that maximizes texture (entropy) in the center region
	cx0, cy0, cx1, cy1 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
	best_k, best_score = 0, -1.0
	for k in range(K):
		mask_k = (label_full == k).astype(np.uint8) * 255
		# Score: entropy median inside center region + overall area near center
		center_patch = ent_norm[cy0:cy1, cx0:cx1]
		center_mask = mask_k[cy0:cy1, cx0:cx1]
		if center_mask.size == 0:
			continue
		ent_score = float(np.median(center_patch[center_mask > 0])) if np.any(center_mask > 0) else 0.0
		area_score = float(np.sum(center_mask > 0)) / center_mask.size
		score = ent_score * 0.7 + area_score * 0.3
		if score > best_score:
			best_score = score
			best_k = k

	fabric_mask = ((label_full == best_k).astype(np.uint8) * 255)
	fabric_mask = cv2.morphologyEx(fabric_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
	fabric_mask = cv2.morphologyEx(fabric_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

	# Combine texture and color priors to refine fabric region
	combined = cv2.bitwise_and(fabric_mask, th_ent)
	combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

	cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not cnts:
		return img_rgb, np.s_[:], combined, gray
	# Prefer components intersecting the image center
	center = (w // 2, h // 2)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	chosen = cnts[0]
	for c in cnts:
		if cv2.pointPolygonTest(c, center, False) >= 0:
			chosen = c
			break
	x, y, w2, h2 = cv2.boundingRect(chosen)
	crop_slice = np.s_[y : y + h2, x : x + w2]

	# Decide cropping behavior
	area_ratio = (w2 * h2) / float(w * h)
	center_mask = combined[cy0:cy1, cx0:cx1] > 0
	center_cov = float(np.sum(center_mask)) / center_mask.size if center_mask.size else 0.0
	# Reuse best_score from k-means scoring
	combined_score = best_score

	if mode == "never":
		return img_rgb, np.s_[:], combined, gray
	if mode == "auto":
		# Safe thresholds: avoid over-cropping if region is too small or barely central
		if area_ratio < 0.25 or center_cov < 0.20 or combined_score < 25.0:
			return img_rgb, np.s_[:], combined, gray
	# mode == "always" or auto with acceptable ROI
	return img_rgb[crop_slice].copy(), crop_slice, combined, gray


def gabor_energy_map(gray, frequencies=(0.1, 0.2, 0.3), orientations=8):
	h, w = gray.shape
	gray_f = gray.astype(np.float32) / 255.0
	energy = np.zeros((h, w), dtype=np.float32)
	for f in frequencies:
		for k in range(orientations):
			theta = (np.pi * k) / orientations
			filt_real, filt_imag = gabor(gray_f, frequency=f, theta=theta)
			energy += (filt_real**2 + filt_imag**2).astype(np.float32)
	energy = cv2.GaussianBlur(energy, (0, 0), 1.0)
	energy_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return energy, energy_norm


def detect_defects_from_energy(energy):
	bg = cv2.medianBlur(energy, 21)
	diff = cv2.subtract(energy, bg)
	diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	_, mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
	# Remove tiny components
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cleaned = np.zeros_like(mask)
	for c in cnts:
		if cv2.contourArea(c) >= 50:
			cv2.drawContours(cleaned, [c], -1, 255, -1)
	return cleaned, diff_norm, bg


def overlay_defects(img_rgb, mask):
	overlay = img_rgb.copy()
	red = np.zeros_like(overlay)
	red[:, :, 0] = 255
	alpha = 0.35
	m3 = np.repeat((mask > 0)[..., None], 3, axis=2)
	overlay[m3] = (alpha * red[m3] + (1 - alpha) * overlay[m3]).astype(np.uint8)
	# Draw bounding boxes
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for c in cnts:
		if cv2.contourArea(c) < 50:
			continue
		x, y, w, h = cv2.boundingRect(c)
		cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
	return overlay


def process_image_img(img_rgb, orientations=8, crop_mode="auto"):
	# Crop fabric region via entropy to focus processing
	cropped, crop_slice, fabric_mask, gray_full = crop_fabric_region(img_rgb, mode=crop_mode)
	gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
	# Gabor energy map (sum of squared responses)
	_, energy_norm = gabor_energy_map(gray_cropped, orientations=orientations)
	# Background suppression + thresholding
	defect_mask, diff_norm, bg_est = detect_defects_from_energy(energy_norm)
	# Restrict defects to fabric region only
	fm_cropped = fabric_mask[crop_slice] if isinstance(crop_slice, tuple) else fabric_mask
	defect_mask = cv2.bitwise_and(defect_mask, fm_cropped)
	# Overlay defects
	overlay = overlay_defects(cropped, defect_mask)
	return {
		"original": img_rgb,
		"crop": cropped,
		"crop_slice": crop_slice,
		"fabric_mask": fabric_mask,
		"gray": gray_cropped,
		"gabor_energy": energy_norm,
		"background": bg_est,
		"anomaly": diff_norm,
		"defect_mask": defect_mask,
		"overlay": overlay,
	}


def process_image(path, crop_mode="auto"):
	original = load_image(path)
	cropped, crop_slice, fabric_mask, gray_full = crop_fabric_region(original, mode=crop_mode)
	gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
	energy_float, energy_norm = gabor_energy_map(gray_cropped)
	defect_mask, diff_norm, bg_est = detect_defects_from_energy(energy_norm)
	# Restrict defects to fabric region only
	fm_cropped = fabric_mask[crop_slice] if isinstance(crop_slice, tuple) else fabric_mask
	defect_mask = cv2.bitwise_and(defect_mask, fm_cropped)
	overlay = overlay_defects(cropped, defect_mask)
	return {
		"path": path,
		"original": original,
		"crop": cropped,
		"crop_slice": crop_slice,
		"fabric_mask": fabric_mask,
		"gray": gray_cropped,
		"gabor_energy": energy_norm,
		"background": bg_est,
		"anomaly": diff_norm,
		"defect_mask": defect_mask,
		"overlay": overlay,
	}


def main():
	st.set_page_config(page_title="Fabric Defect Detection (Gabor)", layout="wide")
	st.title("Fabric Defect Detection using Gabor Filters")
	st.caption(
		"Pipeline: crop fabric → Gabor energy → background suppression → anomaly threshold → defects"
	)

	st.sidebar.header("Parameters")
	orientations = st.sidebar.slider("Orientations", 4, 12, 8, 1)
	crop_option = st.sidebar.selectbox(
		"Cropping",
		["Auto (recommended)", "Always", "Never"],
		index=0,
		help="Control fabric ROI cropping. Auto avoids over-cropping if ROI confidence is low."
	)
	crop_mode = {"Auto (recommended)": "auto", "Always": "always", "Never": "never"}[crop_option]
	uploaded_file = st.file_uploader("Upload a fabric image", type=["jpg", "jpeg", "png"])

	if uploaded_file is None:
		st.info("Upload an image to analyze defects.")
		return

	img = Image.open(uploaded_file).convert("RGB")
	img_rgb = np.array(img)
	st.subheader(uploaded_file.name)
	result = process_image_img(img_rgb, orientations=orientations, crop_mode=crop_mode)

	col1, col2, col3 = st.columns(3)
	with col1:
		st.write("Original")
		st.image(result["original"], use_column_width=True)
		st.write("Cropped Fabric")
		st.image(result["crop"], use_column_width=True)
	with col2:
		st.write("Gabor Energy")
		st.image(result["gabor_energy"], clamp=True, use_column_width=True)
		st.write("Background Estimate")
		st.image(result["background"], clamp=True, use_column_width=True)
	with col3:
		st.write("Anomaly Map")
		st.image(result["anomaly"], clamp=True, use_column_width=True)
		st.write("Defect Overlay")
		st.image(result["overlay"], use_column_width=True)


if __name__ == "__main__":
	main()

