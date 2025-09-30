from dataclasses import dataclass


@dataclass
class RenderConfig:
    """Configuration for 3D rendering parameters."""

    canonical_units: str = "m"
    frustum_near: float = 0.1
    frustum_far: float = 10.0
    axes_length: float = 1.0
    frustum_color: tuple = (0.8, 0.8, 0.8)

    # Unit conversion factors
    _unit_conversion = {
        "m": 1.0,
        "dm": 0.1,
        "cm": 0.01,
        "mm": 0.001,
        "in": 0.0254,
        "ft": 0.3048,
    }

    def scale_factor_from(self, source_units: str) -> float:
        """
        Compute scale factor from source to canonical units.

        Args:
            source_units (str): The units to convert from.

        Returns:
            float: Scale factor to convert from source_units to canonical_units.
        """
        source_factor = self._unit_conversion.get(source_units, 1.0)
        target_factor = self._unit_conversion.get(self.canonical_units, 1.0)
        return source_factor / target_factor
