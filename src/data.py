from typing import Any

import yaml
from pydantic import BaseModel, Field


class Template(BaseModel):
    """
    Represents a measurement template with optional physical properties.

    Templates are reference objects that can be measured against in scenes.
    They may have associated image files and known physical dimensions.

    Attributes:
        id: Unique identifier for the template
        label: Human-readable description or name
        path: File path to associated image or data file
        width: Physical width of the template (in measurement units)
        height: Physical height of the template (in measurement units)
    """

    id: str = Field(..., description="Unique identifier for the template")
    label: str | None = Field(None, description="Human-readable template name")
    path: str | None = Field(None, description="Path to template image/data file")
    width: float | None = Field(
        None, ge=0, description="Physical width in measurement units"
    )
    height: float | None = Field(
        None, ge=0, description="Physical height in measurement units"
    )

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the template.

        Returns:
            Formatted string showing template ID, label, path, and dimensions.

        Example:
            "[T001] Ruler
            * path:   /images/ruler.jpg
            * (w, h): 30.0×2.0"
        """
        # Format path information, handle None case
        path_info: str = (
            f"\n* path:   {self.path}" if self.path else "\n* path:   no path provided"
        )

        # Format dimension information, only show if both width and height exist
        dimensions_info: str = (
            f"\n* (w, h): {self.width}×{self.height}"
            if self.width is not None and self.height is not None
            else "\n* (w, h): dimensions unknown"
        )

        # Combine all information components
        template_info: str = "".join([path_info, dimensions_info])

        # Create main header with ID and label
        header: str = f"[{self.id}] {self.label or 'no label'}"

        return f"{header}{template_info}"


class Distance(BaseModel):
    """
    Represents a distance measurement between two templates in a scene.

    Distances are bidirectional - a distance from A to B is the same as B to A.
    The measurement is stored in the units specified by the parent MeasurementData.

    Attributes:
        from_: Source template ID (aliased from 'from' to avoid Python keyword)
        to: Target template ID
        distance: Measured distance value in measurement units
    """

    from_: str = Field(..., alias="from", description="Source template ID")
    to: str = Field(..., description="Target template ID")
    distance: float = Field(
        ..., ge=0, description="Distance value in measurement units"
    )

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the distance.

        Returns:
            Formatted string showing the bidirectional distance relationship.

        Example:
            "T001 <-> T002 = 15.5"
        """
        return f"{self.from_} <-> {self.to} = {self.distance}"


class Scene(BaseModel):
    """
    Represents a measurement scene containing multiple distance relationships.

    A scene typically corresponds to a single image or measurement context
    where multiple templates are present and distances between them are measured.

    Attributes:
        id: Unique identifier for the scene
        label: Human-readable description of the scene
        path: File path to associated scene image or data
        distances: List of distance measurements within this scene
    """

    id: str = Field(..., description="Unique identifier for the scene")
    label: str | None = Field(None, description="Human-readable scene description")
    path: str | None = Field(None, description="Path to scene image/data file")
    distances: list[Distance] = Field(
        default_factory=list, description="Distance measurements in this scene"
    )

    def get_distance(self, from_id: str, to_id: str) -> Distance | None:
        """
        Retrieve a distance measurement between two templates in this scene.

        This method searches for distances bidirectionally - it will find
        a match regardless of the order of from_id and to_id.

        Args:
            from_id: ID of the first template
            to_id: ID of the second template

        Returns:
            Distance object if found, None otherwise

        Example:
            >>> scene = Scene(id="S001", distances=[...])
            >>> distance = scene.get_distance("T001", "T002")
            >>> if distance:
            ...     print(f"Distance: {distance.distance}")
        """
        # Search through all distances in this scene
        for distance in self.distances:
            # Check both directions since distances are bidirectional
            if (distance.from_ == from_id and distance.to == to_id) or (
                distance.from_ == to_id and distance.to == from_id
            ):
                return distance

        # No matching distance found
        return None

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the scene.

        Returns:
            Formatted string showing scene details and all distance measurements.

        Example:
            "[S001] Laboratory Setup
            * path: /images/lab_scene.jpg
            * distances:
                T001 <-> T002 = 15.5
                T002 <-> T003 = 8.2"
        """
        # Create scene header with ID and label
        header: str = f"[{self.id}] {self.label or 'no label'}"

        # Format path information
        path_info: str = (
            f"* path: {self.path}" if self.path else "* path: no path provided"
        )

        # Format all distance measurements with indentation
        if self.distances:
            distance_strings: list[str] = [str(d) for d in self.distances]
            distances_info: str = "* distances:\n\t" + "\n\t".join(distance_strings)
        else:
            distances_info = "* distances: none"

        return f"{header}\n{path_info}\n{distances_info}"


class MeasurementData(BaseModel):
    """
    Main container for measurement data including templates, scenes, and metadata.

    This class provides the primary interface for managing measurement data,
    including loading from YAML files, validating relationships, and querying
    templates and scenes.

    Attributes:
        unit: Unit of measurement used for all distance values
        templates: List of available measurement templates
        scenes: List of measurement scenes containing distance data
    """

    unit: str = Field(
        ..., description="Unit of measurement (e.g., 'cm', 'mm', 'inches')"
    )
    templates: list[Template] = Field(
        default_factory=list, description="Available measurement templates"
    )
    scenes: list[Scene] = Field(
        default_factory=list, description="Measurement scenes with distance data"
    )

    def __init__(self, **data: Any) -> None:
        """
        Initialize MeasurementData with lookup dictionaries for efficient access.

        Args:
            **data: Keyword arguments passed to parent BaseModel constructor
        """
        super().__init__(**data)

        # Create lookup dictionaries for O(1) access to templates and scenes
        self._template_lookup: dict[str, Template] = {t.id: t for t in self.templates}
        self._scene_lookup: dict[str, Scene] = {s.id: s for s in self.scenes}

    def get_template(self, template_id: str) -> Template | None:
        """
        Retrieve a template by its unique ID.

        Args:
            template_id: Unique identifier of the template to retrieve

        Returns:
            Template object if found, None otherwise
        """
        return self._template_lookup.get(template_id)

    def get_scene(self, scene_id: str) -> Scene | None:
        """
        Retrieve a scene by its unique ID.

        Args:
            scene_id: Unique identifier of the scene to retrieve

        Returns:
            Scene object if found, None otherwise
        """
        return self._scene_lookup.get(scene_id)

    def get_scene_templates(self, scene_id: str) -> list[str] | None:
        """
        Get all template IDs used in a specific scene.

        This method retrieves all templates that are referenced in the distance
        measurements of the specified scene.

        Args:
            scene_id: ID of the scene to analyze

        Returns: template IDs used in the scene, or None if the scene does not exist
        """
        # Validate scene ID
        if scene_id not in self._scene_lookup:
            return None

        # Retrieve the scene object
        scene = self.get_scene(scene_id)
        if not scene:
            return None

        # Use a set to avoid duplicate templates
        templates_in_scene: set[str] = set()
        for distance in scene.distances:
            from_template = self.get_template(distance.from_)
            to_template = self.get_template(distance.to)
            if from_template:
                templates_in_scene.add(from_template.id)
            if to_template:
                templates_in_scene.add(to_template.id)

        return list(templates_in_scene)

    def get_all_scenes(self) -> dict[str, list[str]]:
        """
        Get a mapping of all scenes to the templates they contain.

        This method analyzes all distance measurements in each scene to determine
        which templates are referenced, providing a useful overview of template
        usage across scenes.

        Returns:
            Dictionary mapping scene IDs to lists of template IDs used in each scene

        Example:
            >>> data = MeasurementData(...)
            >>> scene_templates = data.get_all_scenes()
            >>> for scene_id, template_ids in scene_templates.items():
            ...     print(f"Scene {scene_id} uses templates: {template_ids}")
        """
        all_scenes: dict[str, list[str]] = {}

        # Process each scene to extract template usage
        for scene in self.scenes:
            # Use a set to avoid duplicate template IDs
            templates_in_scene: set[str] = set()

            # Extract template IDs from all distance measurements
            for distance in scene.distances:
                # Check if referenced templates actually exist
                from_template = self.get_template(distance.from_)
                to_template = self.get_template(distance.to)

                # Only add template IDs that correspond to existing templates
                if from_template is not None:
                    templates_in_scene.add(from_template.id)
                if to_template is not None:
                    templates_in_scene.add(to_template.id)

            # Convert set to sorted list for consistent output
            all_scenes[scene.id] = sorted(list(templates_in_scene))

        return all_scenes

    def get_distance(self, scene_id: str, from_id: str, to_id: str) -> float | None:
        """
        Get the distance between two templates in a specific scene.

        This method provides a convenient way to query distance measurements
        without needing to first retrieve the scene object.

        Args:
            scene_id: ID of the scene containing the measurement
            from_id: ID of the first template
            to_id: ID of the second template

        Returns:
            Distance value if found, None otherwise

        Raises:
            ValueError: If the specified scene does not exist

        Example:
            >>> data = MeasurementData(...)
            >>> distance = data.get_distance("S001", "T001", "T002")
            >>> if distance is not None:
            ...     print(f"Distance: {distance} {data.unit}")
        """
        # Retrieve the scene, raising an error if not found
        scene = self.get_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene '{scene_id}' not found in measurement data")

        # Search for the distance measurement bidirectionally
        for distance in scene.distances:
            if (distance.from_ == from_id and distance.to == to_id) or (
                distance.from_ == to_id and distance.to == from_id
            ):
                return distance.distance

        # No matching distance found
        return None

    def list_templates(self) -> list[str]:
        """
        Get a list of all template IDs.

        Returns:
            List of template IDs in the order they were defined
        """
        return list(self._template_lookup.keys())

    def list_scenes(self) -> list[str]:
        """
        Get a list of all scene IDs.

        Returns:
            List of scene IDs in the order they were defined
        """
        return list(self._scene_lookup.keys())

    def validate_scene(self, scene_id: str) -> bool:
        """
        Validate that a scene exists and all its distance references are valid.

        This method checks that:
        1. The scene exists in the measurement data
        2. All template IDs referenced in distance measurements correspond to existing templates

        Args:
            scene_id: ID of the scene to validate

        Returns:
            True if the scene is valid, False otherwise

        Example:
            >>> data = MeasurementData(...)
            >>> if data.validate_scene("S001"):
            ...     print("Scene S001 is valid")
            ... else:
            ...     print("Scene S001 has invalid template references")
        """
        # Check if scene exists
        scene = self.get_scene(scene_id)
        if not scene:
            return False

        # Validate all template references in distance measurements
        for distance in scene.distances:
            # Check if both source and target templates exist
            if (
                distance.from_ not in self._template_lookup
                or distance.to not in self._template_lookup
            ):
                return False

        return True

    def print_templates(self) -> None:
        """
        Print a formatted overview of all templates to stdout.

        This method provides a human-readable summary of all templates
        including their properties and relationships.
        """
        print(f"--- Templates ({len(self.templates)} total) ---")
        for template in self.templates:
            print(f"{template}\n")
        print()  # Extra spacing

    def print_scenes(self) -> None:
        """
        Print a formatted overview of all scenes to stdout.

        This method provides a human-readable summary of all scenes
        including their distance measurements.
        """
        print(f"--- Scenes ({len(self.scenes)} total) ---")
        for scene in self.scenes:
            print(f"{scene}\n")
        print()  # Extra spacing

    def print_overview(self) -> None:
        """
        Print a comprehensive overview of the entire measurement dataset.

        This method displays:
        - Measurement unit
        - All templates with their properties
        - All scenes with their distance measurements

        Useful for debugging and getting a quick overview of the data structure.
        """
        print("=== Measurement Data Overview ===\n")
        print(f"[Unit]: {self.unit}\n")
        self.print_templates()
        self.print_scenes()


def load_measurements_from_yaml(path: str) -> MeasurementData:
    """
    Load measurement data from a YAML configuration file.

    This function reads a YAML file containing measurement data and creates
    a validated MeasurementData object. The YAML structure should match
    the MeasurementData model schema.

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
        return MeasurementData(**raw_data)

    except FileNotFoundError:
        raise FileNotFoundError(f"Measurement YAML file not found: {path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in file {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load measurements from {path}: {e}")
