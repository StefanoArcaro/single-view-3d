import cv2
import numpy as np

from src.plotting import visualize_homography
from src.utils import load_rgb


def extract_features(image, method="SIFT"):
    """
    Detect keypoints and compute descriptors.

    Args:
        image: RGB image as numpy array.
        method: Feature detector ('SIFT', 'ORB', etc.).
    Returns:
        keypoints, descriptors
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        raise ValueError(f"Unsupported method: {method}")
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, method="BF", cross_check=True):
    """
    Match feature descriptors between two images.

    Args:
        desc1: Descriptors from image 1.
        desc2: Descriptors from image 2.
        method: 'BF' for BruteForce, 'FLANN' for FLANN matcher.
        cross_check: (BF only) whether to use crossCheck.
    Returns:
        List of matches sorted by distance.
    """
    if method == "BF":
        # Choose norm by descriptor type
        norm = cv2.NORM_L2 if desc1.dtype == np.float32 else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=cross_check)
        matches = matcher.match(desc1, desc2)
    elif method == "FLANN":
        # FLANN parameters for SIFT
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        # Ratio test
        matches = [
            m[0]
            for m in raw_matches
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
        ]
    else:
        raise ValueError(f"Unsupported matcher: {method}")
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def compute_homography(kp1, kp2, matches, ransac_thresh=5.0):
    """
    Compute homography using matched keypoints and calculate reprojection error.

    Args:
        kp1, kp2: Keypoints from image1 and image2.
        matches: List of cv2.DMatch objects.
        ransac_thresh: RANSAC reprojection threshold.
    Returns:
        homography matrix H, mask of inliers, reprojection error (pixels)
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography")

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)

    # Compute reprojection error
    if H is not None:
        # Transform pts1 using homography
        pts1_homogeneous = np.column_stack([pts1, np.ones(len(pts1))])
        transformed_pts = (H @ pts1_homogeneous.T).T

        # Convert back from homogeneous coordinates
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:3]

        # Compute Euclidean distances for all points
        errors = np.linalg.norm(transformed_pts - pts2, axis=1)

        # Calculate mean error for inliers only
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            inlier_errors = errors[inlier_mask]
            reprojection_error = (
                np.mean(inlier_errors) if len(inlier_errors) > 0 else float("inf")
            )
        else:
            reprojection_error = np.mean(errors)
    else:
        reprojection_error = float("inf")

    return H, mask, reprojection_error


def template_match(
    template_path, scene_path, extract_method="SIFT", match_method="BF", plot=True
):
    tpl = load_rgb(template_path)
    img = load_rgb(scene_path)
    kp_t, desc_t = extract_features(tpl, method=extract_method)
    kp_i, desc_i = extract_features(img, method=extract_method)
    matches = match_descriptors(desc_t, desc_i, method=match_method)
    H, mask, reprojection_error = compute_homography(kp_t, kp_i, matches)
    if plot:
        visualize_homography(tpl, img, H, title="Template Matching Result")
    return H, mask, tpl.shape[:2], reprojection_error
