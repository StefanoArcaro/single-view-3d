from typing import Literal

import cv2
import numpy as np

from src.models.scene import Scene
from src.models.template import Template
from src.pipeline.geometry import derive_metric_homography
from src.utils import find_project_root, load_rgb


def extract_features(
    image: np.ndarray,
    method: Literal["SIFT", "ORB"] = "SIFT",
    max_features: int | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """
    Detect keypoints and compute descriptors from an input image.

    This function converts the input RGB image to grayscale and applies the
    specified feature detection algorithm to extract keypoints and their
    corresponding descriptors.

    Args:
        image: Input RGB image as a numpy array with shape (H, W, 3).
        method: Feature detection method to use. Options:
            - "SIFT": Scale-Invariant Feature Transform (float32 descriptors)
            - "ORB": Oriented FAST and Rotated BRIEF (binary descriptors)
        max_features: Optional maximum number of features to extract. If None:
            - SIFT: Uses OpenCV default (unlimited, but practically limited by image)
            - ORB: Uses 2000 (function default)
            If specified, limits keypoints to prevent OpenCV matcher overflow.
            Recommended: <=200,000 to stay under OpenCV's descriptor number limit.

    Returns:
        A tuple containing:
            - keypoints: List of cv2.KeyPoint objects representing detected features
            - descriptors: Array of feature descriptors or None if no features found
                Shape: (n_keypoints, descriptor_length)
                SIFT: float32 array, ORB: uint8 array

    Raises:
        ValueError: If an unsupported feature detection method is specified.

    Example:
        >>> image = cv2.imread('image.jpg')
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> keypoints, descriptors = extract_features(image_rgb, method="SIFT", max_features=10000)
        >>> print(f"Found {len(keypoints)} keypoints")
    """
    # Convert RGB image to grayscale for feature detection
    # Most feature detectors work on single-channel images
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize the appropriate feature detector based on method
    if method == "SIFT":
        # Produces float32 descriptors of length 128
        if max_features is not None:
            detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            detector = cv2.SIFT_create()
    elif method == "ORB":
        # Produces binary descriptors, limit to 2000 features for performance
        n_features = max_features if max_features is not None else 2000
        detector = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError(f"Unsupported feature detection method: {method}")

    # Detect keypoints and compute descriptors simultaneously
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: Literal["BF", "FLANN"] = "BF",
    cross_check: bool = True,
) -> list[cv2.DMatch]:
    """
    Match feature descriptors between two sets using the specified algorithm.

    This function finds correspondences between descriptors from two images
    using either Brute Force or FLANN-based matching with appropriate
    distance metrics and filtering techniques.

    Args:
        desc1: Descriptors from the first image (template).
            Shape: (n_keypoints1, descriptor_length)
        desc2: Descriptors from the second image (scene).
            Shape: (n_keypoints2, descriptor_length)
        method: Matching algorithm to use:
            - "BF": Brute Force matcher with appropriate norm
            - "FLANN": Fast Library for Approximate Nearest Neighbors
        cross_check: Only applicable for BF method. If True, only return
            matches where descriptor i in set1 matches descriptor j
            in set2 AND descriptor j matches descriptor i.

    Returns:
        List of cv2.DMatch objects sorted by distance (best matches first).
        Each DMatch contains:
            - queryIdx: Index in desc1
            - trainIdx: Index in desc2
            - distance: Distance between descriptors (lower is better)

    Raises:
        ValueError: If an unsupported matching method is specified.

    Note:
        - For SIFT descriptors (float32): Uses L2 norm
        - For ORB descriptors (uint8): Uses Hamming distance
        - FLANN method includes ratio test (Lowe's test) with threshold 0.75
    """
    if method == "BF":
        # Choose appropriate distance norm based on descriptor data type
        # Float descriptors (SIFT) use L2 norm, binary descriptors (ORB) use Hamming
        norm = cv2.NORM_L2 if desc1.dtype == np.float32 else cv2.NORM_HAMMING

        # Create Brote Force matcher with cross-checking for additional filtering
        matcher = cv2.BFMatcher(norm, crossCheck=cross_check)

        # Find matches between descriptor sets
        matches = matcher.match(desc1, desc2)

    elif method == "FLANN":
        # FLANN parameters optimized for SIFT-like descriptors
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)

        # Create FLANN matcher
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Get k=2 nearest neighbors for ratio test
        raw_matches: list[list[cv2.DMatch]] = matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test to filter ambiguous matches
        # Keep matches where distance to nearest neighbor is significantly
        # smaller than distance to second nearest neighbor
        matches = [
            m[0]  # Take the best match
            for m in raw_matches
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
        ]
    else:
        raise ValueError(f"Unsupported matching method: {method}")

    # Sort matches by distance (ascending order - best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def compute_homography(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    ransac_thresh: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Compute homography matrix using matched keypoints with RANSAC outlier rejection.

    This function estimates the perspective transformation (homography) that maps
    points from the first image to the second image using the provided matches.
    It also calculates the reprojection error to assess the quality of the fit.

    Args:
        kp1: Keypoints from the template image.
        kp2: Keypoints from the scene image.
        matches: List of cv2.DMatch objects representing correspondences
            between keypoints in kp1 and kp2.
        ransac_thresh: RANSAC reprojection threshold in pixels. Points with
            reprojection error below this threshold are considered inliers.

    Returns:
        A tuple containing:
            - homography: 3x3 homography matrix or None if computation fails.
            - mask: Binary mask indicating inlier matches or None.
            - reprojection_error: Mean reprojection error for inliers in pixels.
                Returns +inf if homography computation fails.

    Raises:
        ValueError: If fewer than 4 matches are provided (minimum for homography).

    Note:
        Homography estimation requires at least 4 point correspondences.
        RANSAC helps reject outliers and provides robust estimation.
        Lower reprojection error indicates better template matching quality.
    """
    # Homography requires minimum 4 point correspondences
    if len(matches) < 4:
        raise ValueError(
            f"Insufficient matches for homography computation: {len(matches)} < 4"
        )

    # Extract 2D coordinates from matched keypoints
    # queryIdx refers to template image keypoints (kp1)
    # trainIdx refers to scene image keypoints (kp2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute homography using RANSAC for robust estimation
    # RANSAC iteratively fits models to random subsets and finds consensus
    homography_result = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)

    # Handle both OpenCV 3.x and 4.x return formats
    if homography_result is not None and len(homography_result) == 2:
        H, mask = homography_result
    else:
        H, mask = None, None

    # Compute reprojection error to assess homography quality
    if H is not None:
        # Transform template points using computed homography
        # Convert to homogeneous coordinates (add column of ones)
        pts1_h = np.column_stack([pts1, np.ones(len(pts1), dtype=np.float32)])

        # Apply homography transformation: H * pts1_h^T
        pts1_h_transformed = (H @ pts1_h.T).T

        # Convert back from homogeneous to Cartesian coordinates
        pts1_transformed = pts1_h_transformed[:, :2] / pts1_h_transformed[:, 2:3]

        # Compute Euclidean distances between transformed and actual points
        errors = np.linalg.norm(pts1_transformed - pts2.astype(np.float64), axis=1)

        # Calculate mean reprojection error for inliers only
        if mask is not None:
            # Convert mask to boolean array for indexing
            inlier_mask = mask.ravel().astype(bool)
            inlier_errors = errors[inlier_mask]

            # Compute mean error for inliers, handling empty case
            reprojection_error = (
                float(np.mean(inlier_errors))
                if len(inlier_errors) > 0
                else float("inf")
            )
        else:
            # No inlier mask available, use all points
            reprojection_error = float(np.mean(errors))
    else:
        # Homography computation failed
        reprojection_error = float("inf")

    return H, mask, reprojection_error


def template_match(
    template_path: str,
    scene_path: str,
    extract_method: Literal["SIFT", "ORB"] = "SIFT",
    match_method: Literal["BF", "FLANN"] = "BF",
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int], float]:
    """
    Perform template matching between a template and scene image.

    This is the main function that orchestrates the complete template matching
    pipeline: loading images, extracting features, matching descriptors, and
    computing the homography transformation.

    Args:
        template_path: File path to the template image.
        scene_path: File path to the scene image where template should be found.
        extract_method: Feature extraction method ("SIFT" or "ORB").
        match_method: Descriptor matching method ("BF" or "FLANN").
        plot: Whether to visualize the matching result using the plotting module.

    Returns:
        A tuple containing:
            - homography: 3x3 transformation matrix or None if matching failed.
            - mask: Inlier mask from RANSAC or None.
            - template_shape: (height, width) of the template image.
            - reprojection_error: Mean reprojection error in pixels.

    Example:
        >>> H, mask, shape, error = template_match(
        ...     'template.jpg',
        ...     'scene.jpg',
        ...     extract_method="SIFT",
        ...     match_method="FLANN"
        ... )
        >>> if H is not None:
        ...     print(f"Template found with error: {error:.2f} pixels")
        ... else:
        ...     print("Template not found in scene")
    """
    # Load template and scene images as RGB arrays
    template = load_rgb(template_path)
    scene = load_rgb(scene_path)

    # Extract features from both images using specified method
    tpl_kp, tpl_desc = extract_features(template, method=extract_method)
    scene_kp, scene_desc = extract_features(scene, method=extract_method)

    # Handle case where feature extraction failed
    if tpl_desc is None or scene_desc is None:
        return None, None, template.shape[:2], float("inf")

    # Match descriptors between template and scene
    matches = match_descriptors(tpl_desc, scene_desc, method=match_method)

    # Compute homography from matched keypoints
    H, mask, error = compute_homography(tpl_kp, scene_kp, matches)

    # Return results including template dimensions for reference
    return H, mask, template.shape[:2], error


def multi_template_match(
    scene: Scene,
    templates: list[Template],
    extract_method: Literal["SIFT", "ORB"] = "SIFT",
    match_method: Literal["BF", "FLANN"] = "BF",
    min_match_count: int = 10,
) -> tuple[np.ndarray, dict]:
    """
    Perform multi-template matching on a scene image for the given templates.

    Args:
        scene (Scene): The scene object containing the image and metadata.
        templates (list[Template]): List of template objects to match against the scene.
        extract_method (Literal['SIFT', 'ORB']): Feature extraction method.
        match_method (Literal['BF', 'FLANN']): Feature matching method.
        min_match_count (int): Minimum number of matches required to consider a valid match.

    Returns:
        np.ndarray: The image size of the scene.
        dict: A dictionary containing, indexed by template id. For each template:
            - 'homography': The computed homography matrix.
            - 'error': The reprojection error.
    """
    # Initialize the results dictionary
    results = {}
    project_root = find_project_root()

    # Load the scene image
    scene_image = load_rgb(project_root / scene.path)

    # Extract its features and descriptors
    scene_keypoints, scene_descriptors = extract_features(
        scene_image, method=extract_method
    )

    # Iterate over each template
    for template in templates:
        # Load the template image
        template_image = load_rgb(project_root / template.path)

        # Extract features and descriptors from the template
        template_keypoints, template_descriptors = extract_features(
            template_image, method=extract_method
        )

        # Match the descriptors between the scene and the template
        matches = match_descriptors(
            template_descriptors, scene_descriptors, method=match_method
        )

        # Check if enough matches are found
        if len(matches) <= min_match_count:
            print(f"Not enough matches found for template {template.id}.")
            continue

        # Compute the homography
        H_px, mask, error = compute_homography(
            template_keypoints, scene_keypoints, matches
        )

        # Derive the metric homography
        H_metric = derive_metric_homography(
            H_px=H_px,
            template_size_px=template_image.shape[:2],
            template_size_metric=(template.height, template.width),
        )

        # Remove the scene keypoints that were used as inliers in the homography computation
        inlier_indices = set()
        for i, match in enumerate(matches):
            if mask[i] == 1:  # This match was an inlier
                inlier_indices.add(
                    match.queryIdx
                )  # queryIdx is the scene keypoint index

        # Keep only the keypoints that weren't used as inliers
        remaining_indices = [
            i for i in range(len(scene_keypoints)) if i not in inlier_indices
        ]
        scene_keypoints = [scene_keypoints[i] for i in remaining_indices]
        scene_descriptors = scene_descriptors[remaining_indices]

        # Store the results
        results[template.id] = {"homography": H_metric, "error": error}

    return scene_image.shape[:2], results
