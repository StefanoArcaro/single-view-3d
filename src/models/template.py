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
