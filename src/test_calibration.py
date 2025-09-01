import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.calibration import CalibrationSimple
from src.geometry import recover_all_poses_from_homography, select_best_solution

# Filter out potential matplotlib warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SyntheticHomographyData:
    """Container for all synthetic homography data and metadata."""

    # Base data (noise-free)
    homographies_clean: list[np.ndarray]
    poses_gt: list[tuple[np.ndarray, np.ndarray]]
    distances: list[float]
    angles: list[float]

    # Generation parameters
    K: np.ndarray
    n_homographies: int


@dataclass
class AnalysisResults:
    """Container for all analysis results."""

    # Calibration results
    focal_length_estimates: dict[float, list[float]] = field(default_factory=dict)
    calibration_success_rates: dict[float, float] = field(default_factory=dict)

    # Pixel error results
    pixel_errors: dict[float, list[float]] = field(default_factory=dict)

    # Noise scale results
    noise_to_signal_ratios: dict[float, list[float]] = field(default_factory=dict)

    # Computed statistics (populated after analysis)
    statistics: dict[str, dict[float, float]] = field(default_factory=dict)

    # Pose estimate proxies (distances and viewing angles)
    distance_estimates: dict[float, list[float]] = field(default_factory=dict)
    angle_estimates: dict[float, list[float]] = field(default_factory=dict)


# ===============================================
# GENERATOR
# ===============================================


class SyntheticDataGenerator:
    """Unified generator for all synthetic homography data needed for calibration analysis."""

    def __init__(self, K: np.ndarray):
        """
        Initialize generator with camera intrinsic matrix.

        Args:
            K: Camera intrinsic matrix (3x3)
        """
        self.K = K.copy()

    def generate_dataset(
        self, n_homographies: int, max_rotation: float, random_seed: int = None
    ) -> SyntheticHomographyData:
        """
        Generate complete synthetic dataset with all noise variants.

        Args:
            n_homographies: Number of homographies to generate
            noise_levels: Array of noise standard deviations to test
            random_seed: Optional random seed for reproducibility

        Returns:
            SyntheticHomographyData containing all generated data
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate clean base data
        homographies_clean = []
        poses_gt = []
        distances = []
        angles = []

        for _ in range(n_homographies):
            # Generate random valid pose
            R_gt, t_gt = self._generate_pose(max_rotation)

            # Compute clean homography
            H_gt = self.K @ np.hstack((R_gt[:, :2], t_gt.reshape(3, 1)))
            H_gt /= H_gt[2, 2]  # Normalize

            # Store results
            homographies_clean.append(H_gt)
            poses_gt.append((R_gt, t_gt))
            distances.append(np.linalg.norm(t_gt))
            angles.append(self._compute_viewing_angle(R_gt))

        # Create and return the dataset
        dataset = SyntheticHomographyData(
            homographies_clean=homographies_clean,
            poses_gt=poses_gt,
            distances=distances,
            angles=angles,
            K=self.K,
            n_homographies=n_homographies,
        )

        self._print_dataset_summary(dataset)
        return dataset

    def _generate_pose(self, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate a random valid camera pose for a frontal planar target."""
        # Random distance between 0.1m and 3.0m
        distance = np.random.uniform(0.1, 3.0)

        # Random tilt angles
        tilt_x = np.random.uniform(-angle, angle) * np.pi / 180
        tilt_y = np.random.uniform(-angle, angle) * np.pi / 180

        # Random in-plane rotation (max 90 degrees)
        rot_z = np.random.uniform(-90, 90) * np.pi / 180

        # Create rotation matrix from Euler angles (ZYX order)
        rotation = R.from_euler("zyx", [rot_z, tilt_y, tilt_x])
        R_gt = rotation.as_matrix()

        # Translation: primarily along Z-axis with small XY offset
        offset_x = np.random.uniform(-0.3, 0.3) * distance
        offset_y = np.random.uniform(-0.3, 0.3) * distance
        t_gt = np.array([offset_x, offset_y, distance])

        return R_gt, t_gt

    def _compute_viewing_angle(self, R: np.ndarray) -> float:
        """Compute the viewing angle (tilt from frontal view) in degrees."""
        camera_z = np.array([0, 0, 1])
        target_normal = R.T @ np.array([0, 0, 1])
        cos_angle = np.dot(camera_z, target_normal)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        return angle

    def _print_dataset_summary(self, dataset: SyntheticHomographyData):
        """Print summary statistics of the generated dataset."""
        print("\n" + "=" * 60)
        print("SYNTHETIC DATASET SUMMARY")
        print("=" * 60)

        print(f"Number of homographies: {dataset.n_homographies}")
        print(f"Camera focal length: {dataset.K[0,0]:.1f}")
        print(f"Principal point: ({dataset.K[0,2]:.1f}, {dataset.K[1,2]:.1f})")

        print("Scene Statistics")
        print(
            f"\tDistance range: {np.min(dataset.distances):.2f} - {np.max(dataset.distances):.2f} m"
        )
        print(
            f"\tMean distance: {np.mean(dataset.distances):.2f} ± {np.std(dataset.distances):.2f} m"
        )
        print(
            f"\tViewing angle range: {np.min(dataset.angles):.1f}° - {np.max(dataset.angles):.1f}°"
        )
        print(
            f"\tMean viewing angle: {np.mean(dataset.angles):.1f}° ± {np.std(dataset.angles):.1f}°\n"
        )

        # Analyze homography element statistics
        H_elements = np.array([H.flatten()[:8] for H in dataset.homographies_clean])
        print("Clean Homography Statistics")
        print(
            f"\tElement magnitude range: [{np.min(np.abs(H_elements)):.4f}, {np.max(np.abs(H_elements)):.4f}]"
        )
        print(f"\tMean element magnitude: {np.mean(np.abs(H_elements)):.4f}")

        print("\nDataset ready for analysis!")
        print("=" * 60)


# ===============================================
# ANALYZER
# ===============================================


class Analyzer:
    """Unified analyzer class for all homography noise effects."""

    def __init__(
        self,
        dataset: SyntheticHomographyData,
        n_trials: int,
        noise_levels: np.ndarray,
    ) -> None:
        """
        Initialize the analyzer with the synthetic dataset.

        Args:
            dataset: SyntheticHomographyData instance
            n_trials: Number of trials to run for each noise level
            noise_levels: Array of noise levels to test
        """
        self.dataset = dataset
        self.n_trials = n_trials
        self.noise_levels = noise_levels

        # Derive image size from the principal point (assumed to be in the center)
        cx, cy = dataset.K[0, 2], dataset.K[1, 2]
        self.principal_point = (cx, cy)
        self.image_size = (int(cx * 2), int(cy * 2))

        # Initialize results container
        self.results = AnalysisResults()

        print("Analyzer initialized:")
        print(f"\tDataset: {dataset.n_homographies} homographies")
        print(f"\tTrials per level: {n_trials}")
        print(f"\tNoise levels: {noise_levels}")
        print(f"\tImage size: {self.image_size}")

    def run_all_analyses(self) -> AnalysisResults:
        """
        Run all analyses on the dataset.

        Returns:
            AnalysisResults containing all computed results
        """
        print("=" * 60)
        print("RUNNING ANALYSIS")
        print("=" * 60)

        # Process each noise level
        for i, sigma in enumerate(self.noise_levels):
            print(
                f"\nProcessing noise level σ = {sigma:.3f} ({i+1}/{len(self.noise_levels)})"
            )
            print("-" * 40)

            # Initialize storage for this noise level
            focal_estimates = []
            pixel_error_list = []
            noise_ratios = []
            distance_estimates = [0, 0, 0]
            angle_estimates = [0, 0, 0]
            calibration_failures = 0
            pixel_error_failures = 0

            # Run N trials for this noise level
            for trial in range(self.n_trials):
                try:
                    # Get homographies for this noise level
                    homographies_clean = self.dataset.homographies_clean

                    # Add noise to the homographies
                    if sigma == 0.0:
                        homographies = homographies_clean
                    else:
                        homographies = [
                            self._add_homography_noise(H, sigma)
                            for H in homographies_clean
                        ]

                    # Run calibration analysis
                    focal_est, cal_success = self._run_calibration_trial(homographies)
                    if cal_success:
                        focal_estimates.append(focal_est)
                    else:
                        calibration_failures += 1

                    # Run pixel error analysis (use first homography as representative)
                    pixel_err, pix_success = self._run_pixel_error_trial(
                        homographies, sigma
                    )
                    if pix_success:
                        pixel_error_list.extend(pixel_err)
                    else:
                        pixel_error_failures += 1

                    # Run noise scale analysis
                    ratios = self._run_noise_scale_trial(homographies)
                    noise_ratios.extend(ratios)

                    # Recover the pose estimate proxies
                    distances, angles = self._run_pose_estimation_trial(homographies)

                    distance_estimates += distances
                    angle_estimates += angles

                except Exception as e:
                    # Log the error but continue
                    if trial == 0:  # Only print first error to avoid spam
                        print(f"\tTrial error: {str(e)}")
                    continue

            # Store results for this noise level
            self.results.focal_length_estimates[sigma] = focal_estimates
            self.results.pixel_errors[sigma] = pixel_error_list
            self.results.noise_to_signal_ratios[sigma] = noise_ratios
            self.results.distance_estimates[sigma] = distance_estimates / self.n_trials
            self.results.angle_estimates[sigma] = angle_estimates / self.n_trials

            # Compute success rates
            cal_success_rate = (
                (self.n_trials - calibration_failures) / self.n_trials * 100
            )
            self.results.calibration_success_rates[sigma] = cal_success_rate

            # Print summary for this noise level
            self._print_noise_level_summary(
                focal_estimates,
                pixel_error_list,
                cal_success_rate,
                pixel_error_failures,
            )

        # Compute aggregate statistics
        self._compute_statistics()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return self.results

    def _add_homography_noise(self, H: np.ndarray, sigma: float) -> np.ndarray:
        """Add Gaussian noise to homography matrix."""
        # Scale the noise to have it consistent with image resolution
        base_resolution = 480
        sigma_scaled = sigma * (base_resolution / self.principal_point[0])
        noise = np.random.normal(0, sigma_scaled, (3, 3))

        # Do not add noise to H[2,2]
        noise[2, 2] = 0
        return H + noise

    def _run_calibration_trial(
        self, homographies: list[np.ndarray]
    ) -> tuple[float, bool]:
        """
        Run calibration for one trial.

        Returns:
            (focal_length_estimate, success_flag)
        """
        try:
            calibration = CalibrationSimple()
            calibration.add_homographies(homographies)
            K_estimated = calibration.calibrate(self.principal_point)
            focal_est = K_estimated[0, 0]

            # Basic sanity check
            if focal_est > 0 and focal_est < 10000:  # Reasonable focal length range
                return focal_est, True
            else:
                return 0.0, False

        except Exception:
            return 0.0, False

    def _run_pixel_error_trial(
        self, homographies_noisy: list[np.ndarray], sigma: float
    ) -> tuple[list[float], bool]:
        """
        Run pixel error analysis comparing noisy homographies to clean ones.

        Returns:
            (list_of_pixel_errors, success_flag)
        """
        # Create grid of world plane points (assuming calibration target is ~30cm x 30cm)
        # Points on the Z=0 plane in world coordinates
        world_size = 0.3  # 30cm calibration target
        x_grid = np.linspace(-world_size / 2, world_size / 2, 20)  # 20x20 grid
        y_grid = np.linspace(-world_size / 2, world_size / 2, 20)

        points = []
        for x in x_grid:
            for y in y_grid:
                points.append([x, y, 1])
        points = np.array(points).T

        all_pixel_errors = []
        successful_homographies = 0

        # Try multiple homographies (up to 5) and collect errors
        max_homographies = min(5, len(homographies_noisy))

        for i in range(max_homographies):
            try:
                H_clean = self.dataset.homographies_clean[i]
                H_noisy = homographies_noisy[i]

                # Project world points to image coordinates
                q_clean = H_clean @ points
                q_noisy = H_noisy @ points

                # Check for reasonable depth values (positive and not too close to zero)
                if np.any(q_clean[2, :] < 0.01) or np.any(q_noisy[2, :] < 0.01):
                    continue

                # Convert to pixel coordinates
                q_clean_px = q_clean[:2] / q_clean[2:3]
                q_noisy_px = q_noisy[:2] / q_noisy[2:3]

                # Check for reasonable pixel coordinates (within image bounds + margin)
                max_coord = max(self.image_size) * 2
                if np.any(np.abs(q_clean_px) > max_coord) or np.any(
                    np.abs(q_noisy_px) > max_coord
                ):
                    continue

                # Compute pixel errors
                errors = np.linalg.norm(q_noisy_px - q_clean_px, axis=0)

                # Filter out extreme errors (likely numerical issues)
                reasonable_errors = errors[errors < 100]

                if len(reasonable_errors) > 0:
                    all_pixel_errors.extend(reasonable_errors)
                    successful_homographies += 1

            except Exception:
                continue

        # Return results if we got measurements from at least one homography
        if len(all_pixel_errors) > 0 and successful_homographies > 0:
            return all_pixel_errors, True
        else:
            return [], False

    def _run_noise_scale_trial(
        self, homographies_noisy: list[np.ndarray]
    ) -> list[float]:
        """
        Run noise scale analysis for one trial.

        Returns:
            list_of_noise_to_signal_ratios
        """
        noise_ratios = []

        # Compare each noisy homography to its clean counterpart
        for i, H_noisy in enumerate(homographies_noisy):
            H_clean = self.dataset.homographies_clean[i]

            # Get homography elements (excluding H[2,2] which is normalized to 1)
            clean_elements = H_clean.flatten()[:8]
            noisy_elements = H_noisy.flatten()[:8]

            # Compute noise (difference between noisy and clean)
            noise = noisy_elements - clean_elements

            # Compute element-wise noise-to-signal ratios
            # Add small epsilon to avoid division by zero
            ratios = np.abs(noise) / (np.abs(clean_elements) + 1e-10)

            # Add all ratios from this homography
            noise_ratios.extend(ratios)

        return noise_ratios

    def _run_pose_estimation_trial(
        self, homographies_noisy: list[np.ndarray]
    ) -> tuple[list[float], list[float]]:
        """Compute pose estimation proxies (distance and viewing angle) from the given noisy homographies."""
        distance_estimates = []
        angle_estimates = []

        for H in homographies_noisy:
            # Recover the pose
            solutions = recover_all_poses_from_homography(H, self.dataset.K)
            R, t, _ = select_best_solution(solutions, expected_z_positive=True)

            # Compute the distance
            distance_estimates.append(np.linalg.norm(t))

            # Compute the viewing angle
            camera_z = np.array([0, 0, 1])
            target_normal = R.T @ np.array([0, 0, 1])
            cos_angle = np.dot(camera_z, target_normal)
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            angle_estimates.append(angle)

        return np.array(distance_estimates), np.array(angle_estimates)

    def _print_noise_level_summary(
        self,
        focal_estimates: list[float],
        pixel_errors: list[float],
        cal_success_rate: float,
        pixel_failures: int,
    ):
        """Print summary statistics for a completed noise level."""
        if focal_estimates:
            f_mean = np.mean(focal_estimates)
            f_std = np.std(focal_estimates)
            f_gt = self.dataset.K[0, 0]
            error_pct = abs(f_mean - f_gt) / f_gt * 100
            print(
                f"\tCalibration: f = {f_mean:.1f} ± {f_std:.1f} (error: {error_pct:.2f}%, success: {cal_success_rate:.1f}%)"
            )
        else:
            print(
                f"\tCalibration: No successful trials (success: {cal_success_rate:.1f}%)"
            )

        if pixel_errors:
            pix_rmse = np.sqrt(np.mean(np.array(pixel_errors) ** 2))
            print(
                f"\tPixel errors: RMSE = {pix_rmse:.2f}px ({len(pixel_errors)} measurements)"
            )
        else:
            print(
                f"\tPixel errors: No successful measurements ({pixel_failures} failures)"
            )

    def _compute_statistics(self):
        """Compute aggregate statistics from raw results for easy plotting and analysis."""
        # Get sorted noise levels to ensure consistent ordering
        noise_levels = sorted(self.results.focal_length_estimates.keys())
        f_gt = self.dataset.K[0, 0]

        # Initialize statistics dictionary with plotter-friendly arrays
        stats = {
            "noise_levels": noise_levels,
            # Focal length statistics
            "focal_length_mean": [],
            "focal_length_std": [],
            "focal_length_error_pct": [],
            # Pixel error statistics
            "pixel_error_rms": [],
            "pixel_error_mean": [],
            "pixel_error_p95": [],
            # Success rates
            "success_rates": [],
        }

        # Process each noise level
        for sigma in noise_levels:
            # === Focal Length Statistics ===
            focal_estimates = self.results.focal_length_estimates.get(sigma, [])
            if focal_estimates:
                f_mean = np.mean(focal_estimates)
                f_std = np.std(focal_estimates)
                f_error_pct = abs(f_mean - f_gt) / f_gt * 100

                stats["focal_length_mean"].append(f_mean)
                stats["focal_length_std"].append(f_std)
                stats["focal_length_error_pct"].append(f_error_pct)
            else:
                # No successful calibrations - use NaN
                stats["focal_length_mean"].append(np.nan)
                stats["focal_length_std"].append(np.nan)
                stats["focal_length_error_pct"].append(np.nan)

            # === Pixel Error Statistics ===
            pixel_errors = self.results.pixel_errors.get(sigma, [])
            if pixel_errors:
                pixel_array = np.array(pixel_errors)
                px_rms = np.sqrt(np.mean(pixel_array**2))
                px_mean = np.mean(pixel_array)
                px_p95 = np.percentile(pixel_array, 95)

                stats["pixel_error_rms"].append(px_rms)
                stats["pixel_error_mean"].append(px_mean)
                stats["pixel_error_p95"].append(px_p95)
            else:
                # No pixel error measurements
                stats["pixel_error_rms"].append(np.nan)
                stats["pixel_error_mean"].append(np.nan)
                stats["pixel_error_p95"].append(np.nan)

            # === Success Rates ===
            success_rate = self.results.calibration_success_rates.get(sigma, 0.0)
            stats["success_rates"].append(success_rate)

        # Convert lists to numpy arrays for easier plotting
        for key in stats:
            if key != "focal_error_vs_pixel_error":  # Keep this as list of tuples
                stats[key] = np.array(stats[key])

        # Store in results
        self.results.statistics = stats

        f_range = (
            np.nanmin(stats["focal_length_mean"]),
            np.nanmax(stats["focal_length_mean"]),
        )
        p_err_range = (
            np.nanmin(stats["pixel_error_rms"]),
            np.nanmax(stats["pixel_error_rms"]),
        )
        succ_range = (
            np.nanmin(stats["success_rates"]),
            np.nanmax(stats["success_rates"]),
        )

        # Print summary
        print("\nStatistics Summary:")
        print(f"\tNoise levels processed: {len(noise_levels)}")
        print(f"\tFocal length range: {f_range[0]:.1f} - {f_range[1]:.1f}")
        print(f"\tMax focal error: {np.nanmax(stats['focal_length_error_pct']):.1f}%")
        print(f"\tPixel error range: {p_err_range[0]:.2f} - {p_err_range[1]:.2f} px")
        print(f"\tSuccess rate range: {succ_range[0]:.1f}% - {succ_range[1]:.1f}%")


# ===============================================
# PLOTTER
# ===============================================


class Plotter:
    """Simple plotter for AnalysisResults with thesis-ready output."""

    def __init__(
        self,
        dataset: SyntheticHomographyData,
        results: AnalysisResults,
        output_dir: str = "figures",
    ):
        """
        Initialize plotter with analysis results.

        Args:
            results: AnalysisResults instance
            output_dir: Directory to save figures
        """
        self.dataset = dataset
        self.results = results
        self.stats = results.statistics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Styling
        plt.style.use("default")
        self.colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F1C40F", "#9B59B6", "#E67E22"]
        self.figsize = (6, 5.5)

    def plot_all(self):
        """Generate all plots."""
        self.plot_focal_length()
        self.plot_pixel_errors()
        self.plot_distance_errors()
        self.plot_angle_errors()

    def plot_focal_length(self):
        """Plot focal length estimates with error bars and relative error."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 2])

        noise_levels = self.stats["noise_levels"]
        f_mean = self.stats["focal_length_mean"]
        f_std = self.stats["focal_length_std"]
        f_gt = self.dataset.K[0, 0]

        # Remove NaN values for plotting
        valid_mask = ~np.isnan(f_mean)
        if not np.any(valid_mask):
            print("Warning: No valid focal length data to plot")
            return

        x_valid = noise_levels[valid_mask]
        f_mean_valid = f_mean[valid_mask]
        f_std_valid = f_std[valid_mask]

        # Top plot: Focal length estimates
        ax1.errorbar(
            x_valid,
            f_mean_valid,
            yerr=f_std_valid,
            marker="o",
            capsize=4,
            capthick=1.5,
            linewidth=2,
            color=self.colors[0],
            markersize=5,
            alpha=0.8,
        )
        ax1.axhline(
            y=f_gt,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Ground Truth ({f_gt:.0f})",
        )
        ax1.set_ylabel("Estimated Focal Length")
        ax1.set_title("Focal Length Estimation vs Noise Level")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Bottom plot: Relative error
        error_pct = self.stats["focal_length_error_pct"][valid_mask]
        ax2.plot(
            x_valid,
            error_pct,
            "o-",
            linewidth=2,
            markersize=5,
            color=self.colors[1],
            alpha=0.8,
        )
        ax2.set_xlabel("Noise Level (σ)")
        ax2.set_ylabel("Absolute Error (%)")
        ax2.set_title("Focal Length Estimation Error")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_pixel_errors(self):
        """Plot pixel error statistics."""
        fig, ax = plt.subplots(figsize=self.figsize)

        noise_levels = self.stats["noise_levels"]
        rms_errors = self.stats["pixel_error_rms"]
        mean_errors = self.stats["pixel_error_mean"]

        # Remove NaN values
        valid_mask = ~np.isnan(rms_errors)
        if not np.any(valid_mask):
            print("Warning: No valid pixel error data to plot")
            return

        x_valid = noise_levels[valid_mask]

        ax.plot(
            x_valid,
            rms_errors[valid_mask],
            "o-",
            label="RMS Error",
            linewidth=2,
            markersize=5,
            color=self.colors[0],
            alpha=0.8,
        )
        ax.plot(
            x_valid,
            mean_errors[valid_mask],
            "s-",
            label="Mean Error",
            linewidth=2,
            markersize=5,
            color=self.colors[1],
            alpha=0.8,
        )

        ax.set_xlabel("Noise Level (σ)")
        ax.set_ylabel("Reprojection Error (px)")
        ax.set_title("Reprojection Error vs Noise Level")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_distance_errors(self):
        """Plot mean distance error percentages vs noise level (with std shading)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        noise_levels = sorted(self.results.distance_estimates.keys())
        distance_gt = np.array(self.dataset.distances)

        mean_errors = []
        std_errors = []

        for sigma in noise_levels:
            estimates = np.array(self.results.distance_estimates[sigma])

            # Ensure lengths match
            if len(estimates) != len(distance_gt):
                print(f"Warning: mismatch in length for sigma={sigma}")
                continue

            errors_pct = np.abs(estimates - distance_gt) / distance_gt * 100
            mean_errors.append(np.nanmean(errors_pct))
            std_errors.append(np.nanstd(errors_pct))

        noise_levels = np.array(noise_levels)
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)

        # Plot mean line
        ax.plot(
            noise_levels,
            mean_errors,
            "o-",
            linewidth=2,
            markersize=5,
            color=self.colors[0],
            alpha=0.9,
            label="Distance Error (mean ± std)",
        )

        # Shaded error band (std)
        ax.fill_between(
            noise_levels,
            np.maximum(mean_errors - std_errors, 0),
            mean_errors + std_errors,
            color=self.colors[0],
            alpha=0.2,
        )

        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("Distance Error (%)", fontsize=14)
        ax.set_title("Mean Distance Error vs Noise Level", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_angle_errors(self):
        """Plot mean angle error percentages vs noise level (with std shading, clipped at 0)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        noise_levels = sorted(self.results.angle_estimates.keys())
        angle_gt = np.array(self.dataset.angles)

        mean_errors = []
        std_errors = []

        for sigma in noise_levels:
            estimates = np.array(self.results.angle_estimates[sigma])

            # Ensure lengths match
            if len(estimates) != len(angle_gt):
                print(f"Warning: mismatch in length for sigma={sigma}")
                continue

            errors_pct = np.abs(estimates - angle_gt) / angle_gt * 100
            mean_errors.append(np.nanmean(errors_pct))
            std_errors.append(np.nanstd(errors_pct))

        noise_levels = np.array(noise_levels)
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)

        # Plot mean line
        ax.plot(
            noise_levels,
            mean_errors,
            "o-",
            linewidth=2,
            markersize=5,
            color=self.colors[1],  # different color from distance for clarity
            alpha=0.9,
            label="Angle Error (mean ± std)",
        )

        # Shaded error band with lower bound clipped at 0
        ax.fill_between(
            noise_levels,
            np.maximum(mean_errors - std_errors, 0),
            mean_errors + std_errors,
            color=self.colors[1],
            alpha=0.2,
        )

        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("Angle Error (%)", fontsize=14)
        ax.set_title("Mean Angle Error vs Noise Level", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def export_figures(self, dpi: int = 300):
        """Export all figures as high-quality PNG files."""
        plt.ioff()  # Turn off interactive mode

        try:
            # Generate and save each plot
            self._save_focal_length(dpi)
            self._save_pixel_errors(dpi)
            self._save_distance_errors(dpi)
            self._save_angle_errors(dpi)

        finally:
            plt.ion()  # Turn interactive mode back on

    def _save_focal_length(self, dpi):
        """Save focal length plot with larger text sizes."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 2])

        noise_levels = self.stats["noise_levels"]
        f_mean = self.stats["focal_length_mean"]
        f_std = self.stats["focal_length_std"]
        f_gt = self.dataset.K[0, 0]

        valid_mask = ~np.isnan(f_mean)
        if not np.any(valid_mask):
            return

        x_valid = noise_levels[valid_mask]
        f_mean_valid = f_mean[valid_mask]
        f_std_valid = f_std[valid_mask]

        # First subplot: focal length with error bars
        ax1.errorbar(
            x_valid,
            f_mean_valid,
            yerr=f_std_valid,
            marker="o",
            capsize=4,
            capthick=1.5,
            linewidth=2,
            color=self.colors[0],
            markersize=5,
            alpha=0.8,
        )
        ax1.axhline(
            y=f_gt,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Ground Truth ({f_gt:.0f})",
        )
        ax1.set_ylabel("Estimated Focal Length", fontsize=14)
        ax1.set_title("Focal Length Estimation vs Noise Level", fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis="both", labelsize=12)

        # Second subplot: focal length error
        error_pct = self.stats["focal_length_error_pct"][valid_mask]
        ax2.plot(
            x_valid,
            error_pct,
            "o-",
            linewidth=2,
            markersize=5,
            color=self.colors[1],
            alpha=0.8,
        )
        ax2.set_xlabel("Noise Level (σ)", fontsize=14)
        ax2.set_ylabel("Absolute Error (%)", fontsize=14)
        ax2.set_title("Focal Length Estimation Error", fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="both", labelsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / "focal_length.png", dpi=dpi, bbox_inches="tight")
        plt.close()

    def _save_pixel_errors(self, dpi):
        """Save pixel errors plot."""
        fig, ax = plt.subplots(figsize=(6, 4))

        noise_levels = self.stats["noise_levels"]
        rms_errors = self.stats["pixel_error_rms"]
        mean_errors = self.stats["pixel_error_mean"]

        valid_mask = ~np.isnan(rms_errors)
        if not np.any(valid_mask):
            return

        x_valid = noise_levels[valid_mask]

        ax.plot(
            x_valid,
            rms_errors[valid_mask],
            "o-",
            label="RMS Error",
            linewidth=2,
            markersize=5,
            color=self.colors[0],
            alpha=0.8,
        )
        ax.plot(
            x_valid,
            mean_errors[valid_mask],
            "s-",
            label="Mean Error",
            linewidth=2,
            markersize=5,
            color=self.colors[1],
            alpha=0.8,
        )

        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("Reprojection Error (px)", fontsize=14)
        ax.set_title("Reprojection Error vs Noise Level", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "pixel_errors.png", dpi=dpi, bbox_inches="tight")
        plt.close()

    def _save_distance_errors(self, dpi):
        """Save mean distance error vs noise level plot with shaded std deviation."""
        fig, ax = plt.subplots(figsize=(6, 4))

        noise_levels = sorted(self.results.distance_estimates.keys())
        distance_gt = np.array(self.dataset.distances)

        mean_errors = []
        std_errors = []

        for sigma in noise_levels:
            estimates = np.array(self.results.distance_estimates[sigma])
            if len(estimates) != len(distance_gt):
                continue
            errors_pct = np.abs(estimates - distance_gt) / distance_gt * 100
            mean_errors.append(np.nanmean(errors_pct))
            std_errors.append(np.nanstd(errors_pct))

        noise_levels = np.array(noise_levels)
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)

        ax.plot(
            noise_levels,
            mean_errors,
            "o-",
            linewidth=2,
            markersize=5,
            color=self.colors[0],
            alpha=0.9,
            label="Distance Error (mean ± std)",
        )

        lower_bound = np.maximum(mean_errors - std_errors, 0)
        upper_bound = mean_errors + std_errors
        ax.fill_between(
            noise_levels, lower_bound, upper_bound, color=self.colors[0], alpha=0.2
        )

        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("Distance Error (%)", fontsize=14)
        ax.set_title("Mean Distance Error vs Noise Level", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "distance_errors.png", dpi=dpi, bbox_inches="tight"
        )
        plt.close()

    def _save_angle_errors(self, dpi):
        """Save mean angle error vs noise level plot with shaded std deviation."""
        fig, ax = plt.subplots(figsize=(6, 4))

        noise_levels = sorted(self.results.angle_estimates.keys())
        angle_gt = np.array(self.dataset.angles)

        mean_errors = []
        std_errors = []

        for sigma in noise_levels:
            estimates = np.array(self.results.angle_estimates[sigma])
            if len(estimates) != len(angle_gt):
                continue
            errors_pct = np.abs(estimates - angle_gt) / angle_gt * 100
            mean_errors.append(np.nanmean(errors_pct))
            std_errors.append(np.nanstd(errors_pct))

        noise_levels = np.array(noise_levels)
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)

        ax.plot(
            noise_levels,
            mean_errors,
            "o-",
            linewidth=2,
            markersize=5,
            color=self.colors[1],
            alpha=0.9,
            label="Angle Error (mean ± std)",
        )

        lower_bound = np.maximum(mean_errors - std_errors, 0)
        upper_bound = mean_errors + std_errors
        ax.fill_between(
            noise_levels, lower_bound, upper_bound, color=self.colors[1], alpha=0.2
        )

        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("Angle Error (%)", fontsize=14)
        ax.set_title("Mean Angle Error vs Noise Level", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / "angle_errors.png", dpi=dpi, bbox_inches="tight")
        plt.close()
