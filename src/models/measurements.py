from typing import Any

from pydantic import BaseModel, Field

from src.models.scene import Scene
from src.models.template import Template


class Measurements(BaseModel):
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
        print()

    def print_scenes(self) -> None:
        """
        Print a formatted overview of all scenes to stdout.

        This method provides a human-readable summary of all scenes
        including their distance measurements.
        """
        print(f"--- Scenes ({len(self.scenes)} total) ---")
        for scene in self.scenes:
            print(f"{scene}\n")
        print()

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
