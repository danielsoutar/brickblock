# TODO: Get isort working so we can sort these imports
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from brickblock.objects import Cube, Cuboid, CompositeCube


# TODO: Decide if we want to use classes for this, what details need adding to
# make these transforms useful, etc.
# TODO: Add docstrings
class SpaceStateChange:
    ...


@dataclass
class Addition(SpaceStateChange):
    timestep_id: int
    name: str | None


@dataclass
class Mutation(SpaceStateChange):
    name: str | None
    primitive_id: int | None
    timestep_id: int | None
    scene_id: int | None
    subject: np.ndarray | tuple[dict[str, Any], dict[str, Any]]


@dataclass
class Deletion(SpaceStateChange):
    timestep_id: int
    name: str | None


class Space:
    """
    Representation of a 3D cartesian coordinate space, which tracks its state
    over time.

    This class contains geometric objects for plotting, and acts as a wrapper
    over the visualisation library.
    """

    # TODO: Clarify dimensions for things being HWD or XYZ (or a mix).
    dims: np.ndarray
    mean: np.ndarray
    total: np.ndarray
    num_objs: int
    primitive_counter: int
    time_step: int
    scene_counter: int
    # TODO: Should these be classes?
    cuboid_coordinates: np.ndarray
    cuboid_visual_metadata: dict[str, list]
    cuboid_index: dict[int, dict[int, list[int]]]
    cuboid_names: dict[str, list[int]]
    changelog: list[SpaceStateChange]

    def __init__(self) -> None:
        self.dims = np.zeros((3, 2))
        self.mean = np.zeros((3, 1))
        self.total = np.zeros((3, 1))
        self.num_objs = 0
        self.primitive_counter = 0
        self.time_step = 0
        self.scene_counter = 0
        self.cuboid_coordinates = np.zeros((10, 6, 4, 3))
        self.cuboid_visual_metadata = {}
        self.cuboid_index = {}
        self.cuboid_names = {}
        self.changelog = []

    def add_cube(self, cube: Cube) -> None:
        """
        TODO: Fill in
        """
        primitive_id = self._add_cuboid_primitive(cube)
        self._add_name(cube.name, [primitive_id])
        self.num_objs += 1
        self.changelog.append(Addition(self.time_step, None))
        self.time_step += 1

    def add_cuboid(self, cuboid: Cuboid) -> None:
        primitive_id = self._add_cuboid_primitive(cuboid)
        self._add_name(cuboid.name, [primitive_id])
        self.num_objs += 1
        self.changelog.append(Addition(self.time_step, None))
        self.time_step += 1

    # TODO: Rather than adding individual cubes, this should be a single call
    # and leverage the provided data better by direct insertion.
    def add_composite(self, composite: CompositeCube) -> None:
        """
        TODO: Fill in
        """
        num_cubes = composite.faces.shape[0]

        primitive_ids = []

        for i in range(num_cubes):
            cube_base_point_idx = (i, 0, 0)
            # Swap the axes around here - otherwise you will get double-swapping
            # of the dimensions.
            base_vector = composite.faces[cube_base_point_idx]
            w, d, h = base_vector
            cube = Cube(
                np.array([h, w, d]),
                scale=1.0,
                facecolor=composite.facecolor,
                linewidth=composite.linewidth,
                edgecolor=composite.edgecolor,
                alpha=composite.alpha,
            )
            primitive_ids.append(self._add_cuboid_primitive(cube))

        self._add_name(composite.name, primitive_ids)

        self.changelog.append(Addition(self.time_step, None))
        self.num_objs += 1
        self.time_step += 1

    def _add_cuboid_primitive(self, cuboid: Cube | Cuboid) -> None:
        """
        TODO: Fill in.
        """
        cuboid_bounding_box = cuboid.get_bounding_box()
        cuboid_mean = np.mean(cuboid.points(), axis=0).reshape((3, 1))

        self.total += cuboid_mean

        self.mean = self.total / (self.primitive_counter + 1)

        if self.primitive_counter == 0:
            dim = cuboid_bounding_box
        else:
            # Since there are multiple objects, ensure the resulting dimensions
            # of the surrounding box are centred around the mean.
            dim = np.array(
                [
                    [
                        min(self.dims[i][0], cuboid_bounding_box[i][0]),
                        max(self.dims[i][1], cuboid_bounding_box[i][1]),
                    ]
                    for i in range(len(cuboid_bounding_box))
                ]
            ).reshape((3, 2))

        self.dims = dim

        current_no_of_entries = self.cuboid_coordinates.shape[0]
        if self.primitive_counter >= current_no_of_entries:
            # refcheck set to False since this avoids issues with the debugger
            # referencing the array!
            self.cuboid_coordinates.resize(
                (2 * current_no_of_entries, *self.cuboid_coordinates.shape[1:]),
                refcheck=False,
            )

        self.cuboid_coordinates[self.primitive_counter] = cuboid.faces
        for key, value in cuboid.get_visual_metadata().items():
            if key in self.cuboid_visual_metadata.keys():
                self.cuboid_visual_metadata[key].append(value)
            else:
                self.cuboid_visual_metadata[key] = [value]

        def add_key_to_nested_dict(d, keys):
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            if keys[-1] not in d:
                d[keys[-1]] = []

        keys = [self.scene_counter, self.time_step]
        add_key_to_nested_dict(self.cuboid_index, keys)
        self.cuboid_index[self.scene_counter][self.time_step].append(
            self.primitive_counter
        )

        primitive_id = self.primitive_counter
        self.primitive_counter += 1

        return primitive_id

    def _add_name(self, name: str | None, primitive_ids: list[int]) -> None:
        if name is not None:
            if name in self.cuboid_names.keys():
                raise Exception(
                    f"There already exists an object with name {name}."
                )
            self.cuboid_names[name] = primitive_ids

    def snapshot(self) -> None:
        """
        TODO: Fill in
        """
        if self.scene_counter not in self.cuboid_index.keys():
            raise Exception(
                "A snapshot must include at least one addition, mutation, or "
                "deletion in the given scene."
            )
        self.scene_counter += 1

    # TODO: Decide whether passing the Axes or having it be fully constructed by
    # brickblock is a good idea.
    # TODO: It seems controlling the azimuth and elevation parameters (which are
    # handily configurable!) is what you need for adjusting the camera.
    # TODO: plt.show shows each figure generated by render(), rather than only
    # the last one (though it shows the last one first). Can this be fixed?
    def render(self) -> tuple[plt.Figure, plt.Axes]:
        """
        TODO: Fill in
        """
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None
        )
        ax = fig.add_subplot(111, projection="3d")
        # Remove everything except the objects to display.
        ax.set_axis_off()

        for scene_id in sorted(self.cuboid_index.keys()):
            timesteps = sorted(self.cuboid_index[scene_id].keys())
            for timestep_id in timesteps:
                # Retrieve the object(s) from the index.
                primitive_ids = self.cuboid_index[scene_id][timestep_id]

                if len(primitive_ids) == 1:
                    ax = self._populate_ax_with_primitive(ax, primitive_ids[0])
                else:
                    ax = self._populate_ax_with_composite(ax, primitive_ids)

        return fig, ax

    def _populate_ax_with_primitive(
        self,
        ax: plt.Axes,
        primitive_id: int,
    ) -> plt.Axes:
        """
        TODO: Fill in
        """
        # Create the object for matplotlib ingestion.
        matplotlib_like_cube = Poly3DCollection(
            self.cuboid_coordinates[primitive_id]
        )
        # Set the visual properties first - check if these can be moved
        # into the Poly3DCollection constructor instead.
        visual_properties = {
            k: self.cuboid_visual_metadata[k][primitive_id]
            for k in self.cuboid_visual_metadata.keys()
        }
        matplotlib_like_cube.set_facecolor(visual_properties["facecolor"])
        matplotlib_like_cube.set_linewidths(visual_properties["linewidth"])
        matplotlib_like_cube.set_edgecolor(visual_properties["edgecolor"])
        matplotlib_like_cube.set_alpha(visual_properties["alpha"])
        ax.add_collection3d(matplotlib_like_cube)

        return ax

    def _populate_ax_with_composite(
        self, ax: plt.Axes, primitive_ids: list[int]
    ) -> plt.Axes:
        """
        TODO: Fill in
        """
        for primitive_id in primitive_ids:
            ax = self._populate_ax_with_primitive(ax, primitive_id)
        return ax
