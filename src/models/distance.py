from pydantic import BaseModel, Field


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
