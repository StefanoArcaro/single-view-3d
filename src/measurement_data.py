from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class Template(BaseModel):
    id: str
    label: Optional[str] = None
    path: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None

    def __str__(self) -> str:
        path = f"\n* path:   {self.path}" if self.path else "no path provided"
        dims = (
            f"\n* (w, h): {self.width}×{self.height}"
            if self.width and self.height
            else "?"
        )
        info = "".join([path, dims])
        return f"[{self.id}] {self.label or 'no label'}{info}"


class Distance(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    distance: float

    def __str__(self) -> str:
        return f"{self.from_} <-> {self.to} = {self.distance}"


class Scene(BaseModel):
    id: str
    label: Optional[str] = None
    path: Optional[str] = None
    distances: List[Distance]

    def get_distance(self, from_id: str, to_id: str) -> Distance | None:
        for d in self.distances:
            if (d.from_ == from_id and d.to == to_id) or (
                d.from_ == to_id and d.to == from_id
            ):
                return d
        return None

    def __str__(self) -> str:
        header = f"[{self.id}] {self.label or 'no label'}"
        path = f"* path: {self.path}" if self.path else "no path provided"
        dists = "* distances:\n\t" + "\n\t".join(str(d) for d in self.distances)
        return f"{header}\n{path}\n{dists}"


class MeasurementData(BaseModel):
    unit: str
    templates: List[Template]
    scenes: List[Scene]

    def __init__(self, **data):
        super().__init__(**data)
        self._template_lookup = {t.id: t for t in self.templates}
        self._scene_lookup = {s.id: s for s in self.scenes}

    def get_template(self, template_id: str) -> Template:
        return self._template_lookup.get(template_id)

    def get_scene(self, scene_id: str) -> Scene:
        return self._scene_lookup.get(scene_id)

    def get_all_scenes(self):
        """
        Returns a dictionary of all scenes indexed by their IDs.
        The values are lists of the templates used in each scene.
        """
        all_scenes = {}
        for scene in self.scenes:
            templates = set()
            for dist in scene.distances:
                from_obj = self.get_template(dist.from_)
                to_obj = self.get_template(dist.to)
                if from_obj is not None:
                    templates.add(from_obj.id)
                if to_obj is not None:
                    templates.add(to_obj.id)
            all_scenes[scene.id] = list(templates)
        return all_scenes

    def get_distance(self, scene_id: str, from_id: str, to_id: str) -> Optional[float]:
        scene = self.get_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene '{scene_id}' not found")
        for d in scene.distances:
            if (d.from_ == from_id and d.to == to_id) or (
                d.from_ == to_id and d.to == from_id
            ):
                return d.distance
        return None

    def list_templates(self) -> List[str]:
        return list(self._template_lookup.keys())

    def list_scenes(self) -> List[str]:
        return list(self._scene_lookup.keys())

    def validate_scene(self, scene_id: str) -> bool:
        scene = self.get_scene(scene_id)
        if not scene:
            return False
        for dist in scene.distances:
            if (
                dist.from_ not in self._template_lookup
                or dist.to not in self._template_lookup
            ):
                return False
        return True

    def print_templates(self):
        print(f"--- Templates ({len(self.templates)} total) ---")
        for t in self.templates:
            print(f"{t}\n")
        print()

    def print_scenes(self):
        print(f"--- Scenes ({len(self.scenes)} total) ---")
        for s in self.scenes:
            print(f"{s}\n")
        print()

    def print_overview(self):
        print("=== Measurement Data Overview ===\n")
        print(f"[Unit]: {self.unit}\n")
        self.print_templates()
        self.print_scenes()


def load_measurements_from_yaml(path: str) -> MeasurementData:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return MeasurementData(**raw)
