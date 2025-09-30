from pydantic import BaseModel, Field

from src.models.distance import Distance


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
