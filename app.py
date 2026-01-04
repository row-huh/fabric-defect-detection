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


def crop_fabric_region(img_rgb):
	gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	ent = entropy(gray, disk(9)).astype(np.float32)
	ent_norm = cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	_, th = cv2.threshold(ent_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
	th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
	cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not cnts:
		return img_rgb, np.s_[:], th, gray
	c = max(cnts, key=cv2.contourArea)
	x, y, w, h = cv2.boundingRect(c)
	crop_slice = np.s_[y : y + h, x : x + w]
	return img_rgb[crop_slice].copy(), crop_slice, th, gray


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


def process_image_img(img_rgb, orientations=8):
	# Crop fabric region via entropy to focus processing
	cropped, crop_slice, fabric_mask, gray_full = crop_fabric_region(img_rgb)
	gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
	# Gabor energy map (sum of squared responses)
	_, energy_norm = gabor_energy_map(gray_cropped, orientations=orientations)
	# Background suppression + thresholding
	defect_mask, diff_norm, bg_est = detect_defects_from_energy(energy_norm)
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


def process_image(path):
	original = load_image(path)
	cropped, crop_slice, fabric_mask, gray_full = crop_fabric_region(original)
	gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
	energy_float, energy_norm = gabor_energy_map(gray_cropped)
	defect_mask, diff_norm, bg_est = detect_defects_from_energy(energy_norm)
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
	uploaded_file = st.file_uploader("Upload a fabric image", type=["jpg", "jpeg", "png"])

	if uploaded_file is None:
		st.info("Upload an image to analyze defects.")
		return

	img = Image.open(uploaded_file).convert("RGB")
	img_rgb = np.array(img)
	st.subheader(uploaded_file.name)
	result = process_image_img(img_rgb, orientations=orientations)

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

