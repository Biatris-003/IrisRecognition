# from pathlib import Path
# import cv2
# import numpy as np
# from typing import Tuple
# import json
# from .segmentation import find_pupil, find_iris, draw_segmentation_overlay
# from .normalization import rubber_sheet
# from .utils import ensure_dir, next_index, save_images
# from .feature_extraction import encode_iris
# import os

# def preprocess_iris_image(image_path: str) -> np.ndarray:
#     """
#     same preprocessing steps (grayscale → median blur → CLAHE → inpaint → normalize).
    
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.medianBlur(gray, 5)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(blurred)
#     # reflection cleanup
#     reflection_mask = cv2.inRange(enhanced, 240, 255)
#     cleaned = cv2.inpaint(enhanced, reflection_mask, 3, cv2.INPAINT_TELEA)
#     normalized = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized

# def run_full_pipeline(image_path: str, out_root: str = "outputs",
#                       radial_res:int = 64, angular_res:int = 360) -> Tuple[str, str, Tuple[int,int,int,int]]:
#     """
#     Returns (segmented_path, normalized_path, (cx,cy,r_pupil,r_iris))
#     """
#     out_dir = ensure_dir(out_root)

#     # preprocess (for segmentation); keep original BGR for overlay
#     bgr = cv2.imread(image_path)
#     if bgr is None:
#         raise FileNotFoundError(image_path)
#     pre = preprocess_iris_image(image_path)

#     # pupil → iris
#     cx, cy, r_pupil = find_pupil(pre)
#     r_iris = find_iris(pre, cx, cy, r_pupil)

#     # overlay
#     pre_bgr = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
#     overlay = draw_segmentation_overlay(pre_bgr, cx, cy, r_pupil, r_iris)
   

#     # normalize (rubber sheet)
#     norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris, radial_res=radial_res, angular_res=angular_res)
    
#     # feature extraction → IrisCode
#     iris_code = encode_iris(norm)
#     # store_iris_code(iris_code)
   
    
#     # save with incremented index
#     idx = next_index(out_dir)
#     seg_path, nor_path = save_images(idx, overlay, norm, out_dir)
#     return seg_path, nor_path, (cx, cy, r_pupil, r_iris)

# # def store_iris_code(person_id, iris_code, path="iris_codes/iris_codes.json"):
# #     """Save or append an iris code to a JSON database file."""
# #     try:
# #         db = json.load(open(path, "r"))
# #         print("Loaded existing iris code database.")
# #     except FileNotFoundError:
# #         db = {}
# #         print("Created new iris code database.")
# #     db[person_id] = iris_code.tolist()
# #     with open(path, "w") as f:
# #         json.dump(db, f, indent=2)
# #     print("Stored iris code successfully.")



# def store_iris_code(iris_code, path="iris_codes/iris_codes.json", person_id: str | None = None):
#     """
#     Save an iris code to a JSON database file.

#     If `person_id` is provided, use it as the JSON key (e.g. "000").
#     Otherwise an automatic `person_XXX` id will be generated (backwards compatible).

#     Returns the key used to store the code.
#     """
#     # ensure parent dir exists
#     path_p = Path(path)
#     path_p.parent.mkdir(parents=True, exist_ok=True)

#     try:
#         with open(path, "r") as fh:
#             db = json.load(fh)
#     except FileNotFoundError:
#         db = {}
#     except json.JSONDecodeError:
#         # treat empty or invalid JSON file as empty DB
#         print(f"Warning: JSON decode error while reading {path}; starting with an empty database.")
#         db = {}

#     # use provided person_id if available
#     if person_id:
#         key = str(person_id)
#         if key in db:
#             print(f"Warning: overwriting existing iris code for '{key}' in {path}")
#     else:
#         # Generate automatic ID for backwards compatibility
#         if len(db) == 0:
#             next_id = 1
#         else:
#             existing_ids = [int(k.split("_")[1]) for k in db.keys() if k.startswith("person_")]
#             next_id = (max(existing_ids) + 1) if existing_ids else 1
#         key = f"person_{next_id:03d}"

#     db[key] = iris_code.tolist()

#     with open(path, "w") as f:
#         json.dump(db, f, indent=2)

#     print(f"Iris code saved as '{key}' in {path}")
#     return key


# # def build_iris_codes_dataset(dataset_dir: str = "CASIA-Iris-Thousand", iris_codes_path: str = "iris_codes/iris_codes.json") -> dict:
# #     """
# #     Iterate through each person folder in the CASIA-Iris-Thousand dataset, take the first image
# #     sample found for each person, run the pipeline steps (preprocess -> segmentation -> normalization -> encoding)
# #     and store the resulting iris codes in the JSON database using `store_iris_code`.

# #     Returns a mapping of <person_folder_name> -> <assigned_person_id or None on failure>.
# #     """
# #     mapping = {}
# #     dataset_p = Path(dataset_dir)

# #     # ensure iris codes dir exists
# #     iris_codes_file = Path(iris_codes_path)
# #     iris_codes_file.parent.mkdir(parents=True, exist_ok=True)

# #     if not dataset_p.exists() or not dataset_p.is_dir():
# #         raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

# #     # # iterate person folders (sorted for determinism)
# #     # person_dirs = sorted([p for p in dataset_p.iterdir() if p.is_dir()])
# #     # iterate person folders (sorted for determinism)
# #     person_dirs = sorted([p for p in dataset_p.iterdir() if p.is_dir()])[:20]  # <-- limit to first 30

# #     # image file extensions to consider
# #     exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

# #     for person_dir in person_dirs:
# #         person_name = person_dir.name
# #         # only consider the left-eye subfolder
# #         left_dir = person_dir / "L"
# #         if not left_dir.exists() or not left_dir.is_dir():
# #             print(f"Left-eye folder not found for {person_name} at {left_dir}, skipping.")
# #             mapping[person_name] = None
# #             continue
# #         try:
# #             # find first image in folder by sorted order
# #             images = []
# #             for e in exts:
# #                 images.extend(sorted(left_dir.glob(e)))
# #             if not images:
# #                 print(f"No images found for {person_name}, skipping.")
# #                 mapping[person_name] = None
# #                 continue

# #             img_path = str(images[0])

# #             # run steps similar to run_full_pipeline but without saving overlays/normals
# #             pre = preprocess_iris_image(img_path)
# #             cx, cy, r_pupil = find_pupil(pre)
# #             r_iris = find_iris(pre, cx, cy, r_pupil)
# #             norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris, radial_res=64, angular_res=360)
# #             iris_code = encode_iris(norm)

# #             # store and capture assigned id
# #             # store using the CASIA folder name as the key (e.g. '000')
# #             assigned_id = store_iris_code(iris_code, path=str(iris_codes_file), person_id=person_name)
# #             mapping[person_name] = assigned_id
# #             print(f"Processed {person_name}: {img_path} -> {assigned_id}")

# #         except Exception as e:
# #             # keep going on error, but record failure
# #             print(f"Failed to process {person_name} ({person_dir}): {e}")
# #             mapping[person_name] = None

# #     # summary
# #     succeeded = sum(1 for v in mapping.values() if v)
# #     failed = sum(1 for v in mapping.values() if not v)
# #     print(f"Finished processing dataset. Succeeded: {succeeded}, Failed: {failed}. Codes written to {iris_codes_file}")
# #     return mapping

# def build_iris_codes_dataset(dataset_dir: str = "CASIA-Iris-Thousand",
#                              iris_codes_path: str = "iris_codes/iris_codes.json") -> dict:
#     """
#     Build an iris code database for the FIRST 10 people.
#     For each person, use up to the FIRST 6 images from the left-eye folder 'L'.
#     Each image is stored under a unique key: '<person_folder>_<img_index:03d>'.

#     Returns a mapping of <person_folder_name> -> list of stored keys (or [] on failure).
#     """
#     mapping: dict[str, list[str] | None] = {}
#     dataset_p = Path(dataset_dir)

#     # ensure iris codes dir exists
#     iris_codes_file = Path(iris_codes_path)
#     iris_codes_file.parent.mkdir(parents=True, exist_ok=True)

#     if not dataset_p.exists() or not dataset_p.is_dir():
#         raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

#     # FIRST 10 people (sorted)
#     person_dirs = sorted([p for p in dataset_p.iterdir() if p.is_dir()])[:100]

#     exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
#     stored_total = 0

#     for person_dir in person_dirs:
#         person_name = person_dir.name
#         keys_for_person: list[str] = []

#         left_dir = person_dir / "L"
#         if not left_dir.exists() or not left_dir.is_dir():
#             print(f"[skip] Left-eye folder not found for {person_name} at {left_dir}")
#             mapping[person_name] = []
#             continue

#         # collect ALL images in L (sorted), then keep only the first 6
#         images: list[Path] = []
#         for e in exts:
#             images.extend(sorted(left_dir.glob(e)))
#         images = images[:6]  # <-- limit to 6 images per subject

#         if not images:
#             print(f"[skip] No images found for {person_name} in {left_dir}")
#             mapping[person_name] = []
#             continue

#         print(f"[info] {person_name}: processing {len(images)} left-eye images (max 6)")

#         for idx, img_p in enumerate(images, start=0):
#             try:
#                 img_path = str(img_p)
#                 pre = preprocess_iris_image(img_path)
#                 cx, cy, r_pupil = find_pupil(pre)
#                 r_iris = find_iris(pre, cx, cy, r_pupil)
#                 norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris,
#                                        radial_res=64, angular_res=360)
#                 iris_code = encode_iris(norm)

#                 key = f"{person_name}_{idx:03d}"  # unique key per image
#                 stored_key = store_iris_code(iris_code,
#                                              path=str(iris_codes_file),
#                                              person_id=key)
#                 keys_for_person.append(stored_key)
#                 stored_total += 1

#                 print(f"  - stored {stored_key} from {img_path}")

#             except Exception as e:
#                 print(f"  ! failed on {img_p}: {e}")

#         mapping[person_name] = keys_for_person

#     # summary
#     people = len(person_dirs)
#     print(f"Finished. People processed: {people} (first 10). "
#           f"Total images stored: {stored_total}. "
#           f"Codes at {iris_codes_file}")
#     return mapping
 



from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Dict, List
import json
import os

from .segmentation import find_pupil, find_iris, draw_segmentation_overlay
from .normalization import rubber_sheet
from .utils import ensure_dir, next_index, save_images
from .feature_extraction import encode_iris


def preprocess_iris_image(image_path: str) -> np.ndarray:
    """
    Grayscale → median blur → CLAHE → inpaint reflections → normalize [0,255].
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
                      radial_res: int = 64, angular_res: int = 360) -> Tuple[str, str, Tuple[int, int, int, int]]:
    """
    Returns (segmented_path, normalized_path, (cx,cy,r_pupil,r_iris))
    """
    out_dir = ensure_dir(out_root)

    # preprocess (for segmentation)
    pre = preprocess_iris_image(image_path)

    # pupil → iris
    cx, cy, r_pupil = find_pupil(pre)
    r_iris = find_iris(pre, cx, cy, r_pupil)

    # overlay ON PREPROCESSED (so user sees what was segmented)
    pre_bgr = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
    overlay = draw_segmentation_overlay(pre_bgr, cx, cy, r_pupil, r_iris)

    # normalize (rubber sheet)
    norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris,
                           radial_res=radial_res, angular_res=angular_res)

    # feature extraction → IrisCode (side-effect free here)
    _ = encode_iris(norm)

    # save with incremented index
    idx = next_index(out_dir)
    seg_path, nor_path = save_images(idx, overlay, norm, out_dir)
    return seg_path, nor_path, (cx, cy, r_pupil, r_iris)


def store_iris_code(iris_code, path: str = "iris_codes/iris_codes.json", person_id: str | None = None):
    """
    Save an iris code to a JSON database file.

    If `person_id` is provided, use it as JSON key (e.g., "000_L_000").
    Otherwise an automatic person_XXX id is generated (legacy behavior).

    Returns the key used to store the code.
    """
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "r") as fh:
            db = json.load(fh)
    except FileNotFoundError:
        db = {}
    except json.JSONDecodeError:
        print(f"Warning: JSON decode error while reading {path}; starting with an empty database.")
        db = {}

    if person_id:
        key = str(person_id)
        if key in db:
            print(f"Warning: overwriting existing iris code for '{key}' in {path}")
    else:
        if len(db) == 0:
            next_id = 1
        else:
            existing_ids = [int(k.split("_")[1]) for k in db.keys() if k.startswith("person_")]
            next_id = (max(existing_ids) + 1) if existing_ids else 1
        key = f"person_{next_id:03d}"

    db[key] = iris_code.tolist()

    with open(path, "w") as f:
        json.dump(db, f, indent=2)

    print(f"Iris code saved as '{key}' in {path}")
    return key


def build_iris_codes_dataset_per_eye(
    dataset_dir: str = "CASIA-Iris-Thousand",
    train_codes_path: str = "iris_codes/iris_codes_train.json",
    test_codes_path: str = "iris_codes/iris_codes_test.json",
    subjects_limit: int = 100,
    gallery_k_per_eye: int = 8,
    radial_res: int = 64,
    angular_res: int = 360,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Per-eye builder with deterministic gallery/probe split:
      - First `subjects_limit` subjects (sorted).
      - Eyes 'L' and 'R' handled separately.
      - First `gallery_k_per_eye` images -> TRAIN (gallery).
      - Remaining images -> TEST (probe).
    Keys stored as: "<subject>_<eye>_<idx:03d>", e.g., "000_L_000".

    Returns: mapping[subject][eye]['train'|'test'] -> list of stored keys
    """
    mapping: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    dataset_p = Path(dataset_dir)

    Path(train_codes_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_codes_path).parent.mkdir(parents=True, exist_ok=True)

    if not dataset_p.exists() or not dataset_p.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    person_dirs = sorted([p for p in dataset_p.iterdir() if p.is_dir()])[:subjects_limit]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    def collect_sorted_images(folder: Path) -> List[Path]:
        imgs: List[Path] = []
        if folder.exists() and folder.is_dir():
            for e in exts:
                imgs.extend(sorted(folder.glob(e)))
        return imgs

    total_train, total_test = 0, 0

    for person_dir in person_dirs:
        subj = person_dir.name
        mapping[subj] = {"L": {"train": [], "test": []},
                         "R": {"train": [], "test": []}}

        for eye in ("L", "R"):
            eye_dir = person_dir / eye
            imgs = collect_sorted_images(eye_dir)
            if not imgs:
                print(f"[skip] No images for {subj}/{eye}")
                continue

            train_imgs = imgs[:gallery_k_per_eye]
            test_imgs = imgs[gallery_k_per_eye:]
            print(f"[info] {subj}/{eye}: total={len(imgs)}  train={len(train_imgs)}  test={len(test_imgs)}")

            k = 0
            # TRAIN (gallery)
            for img_p in train_imgs:
                try:
                    pre = preprocess_iris_image(str(img_p))
                    cx, cy, r_pupil = find_pupil(pre)
                    r_iris = find_iris(pre, cx, cy, r_pupil)
                    norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris,
                                           radial_res=radial_res, angular_res=angular_res)
                    iris_code = encode_iris(norm)
                    key = f"{subj}_{eye}_{k:03d}"; k += 1
                    stored_key = store_iris_code(iris_code, path=train_codes_path, person_id=key)
                    mapping[subj][eye]["train"].append(stored_key)
                    total_train += 1
                except Exception as e:
                    print(f"  ! [train] {subj}/{eye} {img_p.name}: {e}")

            # TEST (probe)
            for img_p in test_imgs:
                try:
                    pre = preprocess_iris_image(str(img_p))
                    cx, cy, r_pupil = find_pupil(pre)
                    r_iris = find_iris(pre, cx, cy, r_pupil)
                    norm, _ = rubber_sheet(pre, cx, cy, r_pupil, r_iris,
                                           radial_res=radial_res, angular_res=angular_res)
                    iris_code = encode_iris(norm)
                    key = f"{subj}_{eye}_{k:03d}"; k += 1
                    stored_key = store_iris_code(iris_code, path=test_codes_path, person_id=key)
                    mapping[subj][eye]["test"].append(stored_key)
                    total_test += 1
                except Exception as e:
                    print(f"  ! [test]  {subj}/{eye} {img_p.name}: {e}")

    print("Finished per-eye gallery/probe split")
    print(f"  Subjects: {len(person_dirs)}")
    print(f"  Train (gallery): {total_train} -> {train_codes_path}")
    print(f"  Test  (probe)  : {total_test} -> {test_codes_path}")
    return mapping

