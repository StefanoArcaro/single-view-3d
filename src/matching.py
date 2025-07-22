from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.plotting import visualize_homography
from src.utils import load_rgb

# Type aliases for better readability
ImageArray = NDArray[np.uint8]
PointArray = NDArray[np.float32]
DescriptorArray = NDArray[np.float32 | np.uint8]
KeypointList = list[cv2.KeyPoint]
MatchList = list[cv2.DMatch]
HomographyMatrix = NDArray[np.float64]
InlierMask = NDArray[np.uint8]


def extract_features(
    image: ImageArray, method: Literal["SIFT", "ORB"] = "SIFT"
) -> tuple[KeypointList, DescriptorArray | None]:
    """
    Detect keypoints and compute descriptors from an input image.

    This function converts the input RGB image to grayscale and applies the
    specified feature detection algorithm to extract keypoints and their
    corresponding descriptors.

    Args:
        image: Input RGB image as a numpy array with shape (H, W, 3) and dtype uint8.
        method: Feature detection method to use. Options:
            - "SIFT": Scale-Invariant Feature Transform (float32 descriptors)
            - "ORB": Oriented FAST and Rotated BRIEF (binary descriptors)

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
        >>> keypoints, descriptors = extract_features(image_rgb, method="SIFT")
        >>> print(f"Found {len(keypoints)} keypoints")
    """
    # Convert RGB image to grayscale for feature detection
    # Most feature detectors work on single-channel images
    gray: NDArray[np.uint8] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize the appropriate feature detector based on method
    if method == "SIFT":
        # Produces float32 descriptors of length 128
        detector = cv2.SIFT_create()
    elif method == "ORB":
        # Produces binary descriptors, limit to 2000 features for performance
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        raise ValueError(f"Unsupported feature detection method: {method}")

    # Detect keypoints and compute descriptors simultaneously
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_descriptors(
    desc1: DescriptorArray,
    desc2: DescriptorArray,
    method: Literal["BF", "FLANN"] = "BF",
    cross_check: bool = True,
) -> MatchList:
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
        matches: MatchList = matcher.match(desc1, desc2)

    elif method == "FLANN":
        # FLANN parameters optimized for SIFT-like descriptors
        index_params = dict(
            algorithm=1,  # FLANN_INDEX_KDTREE for float descriptors
            trees=5,  # Number of randomized k-d trees
        )
        search_params = dict(
            checks=50  # Number of times trees should be recursively traversed
        )

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
    kp1: KeypointList, kp2: KeypointList, matches: MatchList, ransac_thresh: float = 5.0
) -> tuple[HomographyMatrix | None, InlierMask | None, float]:
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
            - homography: 3x3 homography matrix (float64) or None if computation fails
            - mask: Binary mask indicating inlier matches (uint8) or None
            - reprojection_error: Mean reprojection error for inliers in pixels (float)
                                 Returns inf if homography computation fails

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
    pts1: PointArray = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2: PointArray = np.float32([kp2[m.trainIdx].pt for m in matches])

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
        pts1_homogeneous: NDArray[np.float32] = np.column_stack(
            [pts1, np.ones(len(pts1), dtype=np.float32)]
        )

        # Apply homography transformation: H * pts1_homogeneous^T
        transformed_pts_homogeneous: NDArray[np.float64] = (H @ pts1_homogeneous.T).T

        # Convert back from homogeneous to Cartesian coordinates
        transformed_pts: NDArray[np.float64] = (
            transformed_pts_homogeneous[:, :2] / transformed_pts_homogeneous[:, 2:3]
        )

        # Compute Euclidean distances between transformed and actual points
        errors: NDArray[np.float64] = np.linalg.norm(
            transformed_pts - pts2.astype(np.float64), axis=1
        )

        # Calculate mean reprojection error for inliers only
        if mask is not None:
            # Convert mask to boolean array for indexing
            inlier_mask: NDArray[np.bool_] = mask.ravel().astype(bool)
            inlier_errors: NDArray[np.float64] = errors[inlier_mask]

            # Compute mean error for inliers, handle empty case
            reprojection_error: float = (
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
    plot: bool = True,
) -> tuple[HomographyMatrix | None, InlierMask | None, tuple[int, int], float]:
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
            - homography: 3x3 transformation matrix or None if matching failed
            - mask: Inlier mask from RANSAC or None
            - template_shape: (height, width) of the template image
            - reprojection_error: Mean reprojection error in pixels

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
    template: ImageArray = load_rgb(template_path)
    scene: ImageArray = load_rgb(scene_path)

    # Extract features from both images using specified method
    template_keypoints, template_descriptors = extract_features(
        template, method=extract_method
    )
    scene_keypoints, scene_descriptors = extract_features(scene, method=extract_method)

    # Handle case where feature extraction failed
    if template_descriptors is None or scene_descriptors is None:
        return None, None, template.shape[:2], float("inf")

    # Match descriptors between template and scene
    matches: MatchList = match_descriptors(
        template_descriptors, scene_descriptors, method=match_method
    )

    # Compute homography from matched keypoints
    homography, inlier_mask, reprojection_error = compute_homography(
        template_keypoints, scene_keypoints, matches
    )

    # Optional visualization of the matching result
    if plot and homography is not None:
        visualize_homography(
            template, scene, homography, title="Template Matching Result"
        )

    # Return results including template dimensions for reference
    return homography, inlier_mask, template.shape[:2], reprojection_error


# ============================================================


def template_match_for_calibration(
    template_path: str,
    scene_path: str,
    extract_method: Literal["SIFT", "ORB"] = "SIFT",
    match_method: Literal["BF", "FLANN"] = "BF",
    plot: bool = True,
    min_inliers: int = 10,
    template_real_size: tuple[float, float] = None,
) -> tuple[
    HomographyMatrix | None,
    InlierMask | None,
    tuple[int, int],
    float,
    np.ndarray | None,
    np.ndarray | None,
]:
    """
    Enhanced template matching that also prepares keypoint correspondences for camera calibration.

    This function performs template matching and additionally extracts the inlier keypoint
    correspondences in the format required by CalibrationSimple.refine_intrinsics().

    Args:
        template_path: File path to the template image.
        scene_path: File path to the scene image where template should be found.
        extract_method: Feature extraction method ("SIFT" or "ORB").
        match_method: Descriptor matching method ("BF" or "FLANN").
        plot: Whether to visualize the matching result.
        min_inliers: Minimum number of inliers required for valid result.
        template_real_size: Real-world size of template (width, height) in mm/meters.
                          If None, uses normalized coordinates [0,1] x [0,1].

    Returns:
        A tuple containing:
            - homography: 3x3 transformation matrix or None if matching failed
            - mask: Inlier mask from RANSAC or None
            - template_shape: (height, width) of the template image
            - reprojection_error: Mean reprojection error in pixels
            - keypoint_pairs: Array of shape (N, 4) with [x_scene, y_scene, x_template, y_template]
            - template_points: Array of shape (N, 2) with [X_world, Y_world] coordinates
    """
    # Load template and scene images as RGB arrays
    template: ImageArray = load_rgb(template_path)
    scene: ImageArray = load_rgb(scene_path)

    # Extract features from both images using specified method
    template_keypoints, template_descriptors = extract_features(
        template, method=extract_method
    )
    scene_keypoints, scene_descriptors = extract_features(scene, method=extract_method)

    # Handle case where feature extraction failed
    if template_descriptors is None or scene_descriptors is None:
        return None, None, template.shape[:2], float("inf"), None, None

    # Match descriptors between template and scene
    matches: MatchList = match_descriptors(
        template_descriptors, scene_descriptors, method=match_method
    )

    # Compute homography from matched keypoints
    homography, inlier_mask, reprojection_error = compute_homography(
        template_keypoints, scene_keypoints, matches
    )

    # Initialize return values for calibration data
    keypoint_pairs = None
    template_points = None

    if homography is not None and inlier_mask is not None:
        # Extract inlier matches only
        inlier_matches = [
            matches[i] for i, is_inlier in enumerate(inlier_mask) if is_inlier
        ]

        # Check if we have enough inliers
        if len(inlier_matches) < min_inliers:
            print(
                f"Warning: Only {len(inlier_matches)} inliers found, minimum {min_inliers} required"
            )
            return (
                homography,
                inlier_mask,
                template.shape[:2],
                reprojection_error,
                None,
                None,
            )

        # Extract keypoint coordinates for inlier matches
        template_pts_img = []
        scene_pts_img = []

        for match in inlier_matches:
            # Get keypoint coordinates
            template_kp = template_keypoints[match.queryIdx]
            scene_kp = scene_keypoints[match.trainIdx]

            # Extract (x, y) coordinates
            template_pts_img.append([template_kp.pt[0], template_kp.pt[1]])
            scene_pts_img.append([scene_kp.pt[0], scene_kp.pt[1]])

        template_pts_img = np.array(template_pts_img)
        scene_pts_img = np.array(scene_pts_img)

        # Prepare keypoint_pairs in format [x_scene, y_scene, x_template, y_template]
        keypoint_pairs = np.hstack([scene_pts_img, template_pts_img])

        # Convert template pixel coordinates to world coordinates
        template_h, template_w = template.shape[:2]

        if template_real_size is not None:
            # Use real-world dimensions
            real_width, real_height = template_real_size

            # Convert from pixel coordinates to world coordinates
            # Assuming template coordinate system with origin at top-left
            template_x_world = (template_pts_img[:, 0] / template_w) * real_width
            template_y_world = (template_pts_img[:, 1] / template_h) * real_height

            template_points = np.column_stack([template_x_world, template_y_world])
        else:
            # Use normalized coordinates [0,1] x [0,1]
            template_x_norm = template_pts_img[:, 0] / template_w
            template_y_norm = template_pts_img[:, 1] / template_h

            template_points = np.column_stack([template_x_norm, template_y_norm])

    # Optional visualization of the matching result
    if plot and homography is not None:
        visualize_homography(
            template, scene, homography, title="Template Matching Result"
        )

        # Additional plot showing inlier keypoints if calibration data is available
        if keypoint_pairs is not None:
            print(
                f"Extracted {len(keypoint_pairs)} inlier correspondences for calibration"
            )

    # Return results including calibration data
    return (
        homography,
        inlier_mask,
        template.shape[:2],
        reprojection_error,
        keypoint_pairs,
        template_points,
    )


def batch_template_match_for_calibration(
    template_paths: list[str],
    scene_path: str,
    extract_method: Literal["SIFT", "ORB"] = "SIFT",
    match_method: Literal["BF", "FLANN"] = "BF",
    plot: bool = False,
    min_inliers: int = 10,
    template_real_sizes: list[tuple[float, float]] = None,
) -> tuple[list[HomographyMatrix], list[np.ndarray], list[np.ndarray]]:
    """
    Batch process multiple templates against a single scene for camera calibration.

    Args:
        template_paths: List of paths to template images.
        scene_path: Path to the scene image containing all templates.
        extract_method: Feature extraction method.
        match_method: Descriptor matching method.
        plot: Whether to visualize each matching result.
        min_inliers: Minimum inliers required per template.
        template_real_sizes: List of real-world sizes for each template.
                           If None, uses normalized coordinates for all.

    Returns:
        A tuple containing:
            - valid_homographies: List of homography matrices for successful matches
            - all_keypoint_pairs: List of keypoint correspondence arrays
            - all_template_points: List of template world coordinate arrays
    """
    valid_homographies = []
    all_keypoint_pairs = []
    all_template_points = []

    if template_real_sizes is None:
        template_real_sizes = [None] * len(template_paths)

    print(f"Processing {len(template_paths)} templates against scene...")

    for i, (template_path, real_size) in enumerate(
        zip(template_paths, template_real_sizes)
    ):
        print(f"\nProcessing template {i + 1}/{len(template_paths)}: {template_path}")

        result = template_match_for_calibration(
            template_path=template_path,
            scene_path=scene_path,
            extract_method=extract_method,
            match_method=match_method,
            plot=plot,
            min_inliers=min_inliers,
            template_real_size=real_size,
        )

        homography, mask, shape, error, keypoint_pairs, template_points = result

        if homography is not None and keypoint_pairs is not None:
            print(
                f"✓ Template {i + 1}: {len(keypoint_pairs)} inliers, error: {error:.2f}px"
            )
            valid_homographies.append(homography)
            all_keypoint_pairs.append(keypoint_pairs)
            all_template_points.append(template_points)
        else:
            print(f"✗ Template {i + 1}: Failed to match or insufficient inliers")

    print(
        f"\nSuccessfully processed {len(valid_homographies)}/{len(template_paths)} templates"
    )

    return valid_homographies, all_keypoint_pairs, all_template_points
