from pathlib import Path
import cv2
import numpy as np
from typing import Tuple

from .segmentation import find_pupil, find_iris, draw_segmentation_overlay
from .normalization import rubber_sheet
from .utils import ensure_dir, next_index, save_images

def preprocess_iris_image(image_path: str) -> np.ndarray:
    """
    same preprocessing steps (grayscale → median blur → CLAHE → inpaint → normalize).
    
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    # reflection cleanup
    reflection_mask = cv2.inRange(enhanced, 240, 255)
    cleaned = cv2.inpaint(enhanced, reflection_mask, 3, cv2.INPAINT_TELEA)
    normalized = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def run_full_pipeline(image_path: str, out_root: str = "outputs",
                      radial_res:int = 64, angular_res:int = 360) -> Tuple[str, str, Tuple[int,int,int,int]]:
    """
    Returns (segmented_path, normalized_path, (cx,cy,r_pupil,r_iris))
    """
    out_dir = ensure_dir(out_root)

    # preprocess (for segmentation); keep original BGR for overlay
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    pre = preprocess_iris_image(image_path)

    # pupil → iris
    cx, cy, r_pupil = find_pupil(pre)
    r_iris = find_iris(pre, cx, cy, r_pupil)

    # overlay
    overlay = draw_segmentation_overlay(bgr, cx, cy, r_pupil, r_iris)

    # normalize (rubber sheet)
    norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris, radial_res=radial_res, angular_res=angular_res)

    # save with incremented index
    idx = next_index(out_dir)
    seg_path, nor_path = save_images(idx, overlay, norm, out_dir)
    return seg_path, nor_path, (cx, cy, r_pupil, r_iris)
