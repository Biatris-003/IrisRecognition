import cv2
import numpy as np
from typing import Tuple

def _best_circle_by_darkness(gray: np.ndarray, circles: np.ndarray) -> Tuple[int,int,int]:
    """
    Among Hough circles, choose the one whose interior mean intensity is the lowest (darkest).
    Returns (cx, cy, r). Falls back to the first if stats fail.
    """
    if circles is None or len(circles) == 0:
        raise ValueError("No pupil circles found.")
    circles = np.uint16(np.around(circles[0]))
    best = circles[0]
    best_mean = 1e9
    h, w = gray.shape[:2]
    Y, X = np.ogrid[:h, :w]
    for (cx, cy, r) in circles:
        r = int(r)
        mask = (X - cx)**2 + (Y - cy)**2 <= r*r
        if not np.any(mask):
            continue
        m = float(np.mean(gray[mask]))
        if m < best_mean:
            best_mean, best = m, (cx, cy, r)
    return int(best[0]), int(best[1]), int(best[2])

def find_pupil(gray: np.ndarray) -> Tuple[int, int, int]:
    """
    Robust-ish pupil detection:
      1) Invert + blur to boost dark blobs
      2) HoughCircles for candidate circles
      3) Pick darkest interior
    Returns (cx, cy, r)
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (9, 9), 2)
    h, w = gray.shape[:2]
    minR = max(6, min(h, w) // 50)
    maxR = max(min(h, w) // 3, minR + 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w)//8,
        param1=120, param2=18, minRadius=minR, maxRadius=maxR
    )
    cx, cy, r = _best_circle_by_darkness(gray, circles)
    return cx, cy, r

def _circle_points(cx:int, cy:int, r:float, num:int) -> Tuple[np.ndarray, np.ndarray]:
    thetas = np.linspace(0, 2*np.pi, num, endpoint=False)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)
    return xs, ys

def _sample_gradient_on_circle(grad: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Bilinear sample gradient magnitude along given circle points; return mean response.
    """
    h, w = grad.shape[:2]
    # bilinear
    x0 = np.clip(np.floor(xs).astype(int), 0, w-2)
    y0 = np.clip(np.floor(ys).astype(int), 0, h-2)
    dx = xs - x0
    dy = ys - y0
    g00 = grad[y0, x0]
    g01 = grad[y0, x0+1]
    g10 = grad[y0+1, x0]
    g11 = grad[y0+1, x0+1]
    g = (g00*(1-dx)*(1-dy) + g01*dx*(1-dy) + g10*(1-dx)*dy + g11*dx*dy)
    return float(np.mean(g))

def find_iris(gray: np.ndarray, cx:int, cy:int, r_pupil:int) -> int:
    """
    Daugman-like 1D scan: for radii from ~1.5*rp to ~min_dim/2, pick radius with
    maximum mean gradient along the circle.
    Returns iris radius.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    h, w = gray.shape[:2]
    r_min = int(max(r_pupil * 1.5, r_pupil + 8))
    r_max = int(min(h, w) * 0.48)
    if r_min >= r_max:
        r_min = max(r_pupil + 4, r_pupil + 2)
        r_max = max(r_min + 10, r_min + 8)

    best_r = r_min
    best_val = -1.0
    for r in range(r_min, r_max, 1):
        xs, ys = _circle_points(cx, cy, r, num=720)  # 0.5° sampling
        val = _sample_gradient_on_circle(grad, xs, ys)
        if val > best_val:
            best_val, best_r = val, r

    return int(best_r)

def draw_segmentation_overlay(bgr: np.ndarray, cx:int, cy:int, r_pupil:int, r_iris:int) -> np.ndarray:
    out = bgr.copy()
    cv2.circle(out, (cx, cy), r_iris, (255, 0, 0), 2)   # blue iris
    cv2.circle(out, (cx, cy), r_pupil, (0, 255, 0), 2)  # green pupil
    return out
