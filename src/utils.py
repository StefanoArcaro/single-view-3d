import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.models.measurements import Measurements


def find_project_root() -> Path:
    """Find the project root by looking for marker files."""
    current = Path(__file__).resolve()

    # Project root markers
    markers = ["src", "README.md", "pyproject.toml"]

    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    return current.parent


def load_rgb(path: str) -> np.ndarray:
    """
    Load image from path and convert to RGB.

    Args:
        path: The file path to the image.

    Returns:
        The image in RGB format.
    """
    return cv2.imread(path, cv2.IMREAD_COLOR_RGB)


def load_calibration_json(
    filename: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Load camera calibration data from a JSON file.

    Args:
        filename: The file path to the JSON file.

    Returns:
        A tuple containing the camera matrix, distortion coefficients, and image size.
    """
    with open(filename) as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeff = np.array(data["dist_coeff"])
    image_size = tuple(data["image_size"])

    return camera_matrix, dist_coeff, image_size


def load_measurements_from_yaml(path: str) -> Measurements:
    """
    Load measurement data from a YAML configuration file.

    This function reads a YAML file containing measurement data and creates
    a validated Measurements object. The YAML structure should match
    the Measurements model schema.

    Args:
        path: File path to the YAML configuration file

    Returns:
        MeasurementData object populated with data from the YAML file

    Raises:
        FileNotFoundError: If the specified file does not exist
        yaml.YAMLError: If the file contains invalid YAML syntax
        pydantic.ValidationError: If the YAML data doesn't match the expected schema

    Example:
        >>> try:
        ...     data = load_measurements_from_yaml("measurements.yaml")
        ...     print(f"Loaded {len(data.templates)} templates and {len(data.scenes)} scenes")
        ... except Exception as e:
        ...     print(f"Failed to load measurements: {e}")

    Expected YAML structure:
        ```yaml
        unit: "cm"
        templates:
          - id: "T001"
            label: "Ruler"
            path: "/images/ruler.jpg"
            width: 30.0
            height: 2.0
        scenes:
          - id: "S001"
            label: "Test Scene"
            path: "/images/scene1.jpg"
            distances:
              - from: "T001"
                to: "T002"
                distance: 15.5
        ```
    """
    try:
        # Open and parse the YAML file
        with open(path, "r", encoding="utf-8") as file:
            raw_data: dict[str, Any] = yaml.safe_load(file)

        # Create and validate MeasurementData object
        return Measurements(**raw_data)

    except FileNotFoundError:
        raise FileNotFoundError(f"Measurement YAML file not found: {path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in file {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load measurements from {path}: {e}")
