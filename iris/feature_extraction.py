import numpy as np
import cv2

# Feature Extraction (Using Gabor filters)

def gabor_filter_bank(size=(9, 9), sigma=3.0, theta=0, lambd=4.0, gamma=1.0, psi=0):
    """Return real and imaginary Gabor kernels."""
    
    x = np.linspace(-size[0]//2, size[0]//2, size[0])
    y = np.linspace(-size[1]//2, size[1]//2, size[1])
    X, Y = np.meshgrid(x, y)

    # Rotation
    x_prime = X * np.cos(theta) + Y * np.sin(theta)
    y_prime = -X * np.sin(theta) + Y * np.cos(theta)

    gaussian = np.exp(-(x_prime**2 + (gamma**2) * (y_prime**2)) / (2 * sigma**2))

    real = gaussian * np.cos(2 * np.pi * x_prime / lambd + psi)
    imag = gaussian * np.sin(2 * np.pi * x_prime / lambd + psi)

    return real, imag


def sample_points(image_shape, radial_bands=8, angular_sectors=128):
    """
    Divide the normalized iris image (e.g., 64x360) into 8x128 patches.
    Return the center coordinates of each patch.
    """
    rows, cols = image_shape
    r_positions = np.linspace(0, rows - 1, radial_bands + 1, dtype=int)
    theta_positions = np.linspace(0, cols - 1, angular_sectors + 1, dtype=int)

    points = []
    for i in range(radial_bands):
        for j in range(angular_sectors):
            r_center = (r_positions[i] + r_positions[i + 1]) // 2
            theta_center = (theta_positions[j] + theta_positions[j + 1]) // 2
            points.append((r_center, theta_center))
    return points  


def encode_iris(normalized_iris, radial_bands=8, angular_sectors=128):
    """
    Apply Gabor filter at 1024 (r,θ) sampling points.
    Convert phase to 2048-bit IrisCode (2 bits per location).
    """
    gabor_real, gabor_imag = gabor_filter_bank()
    patch_size = gabor_real.shape[0] // 2

    sample_pts = sample_points(normalized_iris.shape, radial_bands, angular_sectors)

    bits = []
    for (r, theta) in sample_pts:
        # Extract small patch around the sampling point
        r1, r2 = r - patch_size, r + patch_size + 1
        t1, t2 = theta - patch_size, theta + patch_size + 1
        patch = normalized_iris[max(r1,0):min(r2,normalized_iris.shape[0]),
                                max(t1,0):min(t2,normalized_iris.shape[1])]

        # If patch is too small near border, skip
        if patch.shape[0] < gabor_real.shape[0] or patch.shape[1] < gabor_real.shape[1]:
            continue

        # Apply Gabor filtering
        real_response = np.sum(patch * gabor_real)
        imag_response = np.sum(patch * gabor_imag)

        # Phase sign: 2 bits per location
        bit1 = 1 if real_response >= 0 else 0
        bit2 = 1 if imag_response >= 0 else 0

        bits.extend([bit1, bit2])

    iris_code = np.array(bits, dtype=np.uint8)
    # Ensure 2048 bits (trim or pad if needed)
    iris_code = iris_code[:2048] if iris_code.size >= 2048 else np.pad(iris_code, (0, 2048 - iris_code.size))
    return iris_code