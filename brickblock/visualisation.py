from typing import Any

import itertools
import math
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def materialise_vertices_for_primitive(
    base: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    w = shape[0] * np.array([1, 0, 0])
    h = shape[1] * np.array([0, 0, 1])
    d = shape[2] * np.array([0, 1, 0])
    # Shorthand convention is to have the 'bottom-left-front' point as
    # the base, with points defining width/height/depth of the cube
    # after (using the left-hand rule).
    # Note: the ordering of points matters.
    points = np.array(
        [
            # bottom-left-front
            base,
            # bottom-left-back
            base + d,
            # bottom-right-back
            base + w + d,
            # bottom-right-front
            base + w,
            # top-left-front
            base + h,
            # top-left-back
            base + h + d,
            # top-right-back
            base + h + w + d,
            # top-right-front
            base + h + w,
        ]
    )

    return np.array(
        [
            (points[0], points[1], points[2], points[3]),  # bottom
            (points[0], points[4], points[7], points[3]),  # front face
            (points[0], points[1], points[5], points[4]),  # left face
            (points[3], points[7], points[6], points[2]),  # right face
            (points[1], points[5], points[6], points[2]),  # back face
            (points[4], points[5], points[6], points[7]),  # top
        ]
    ).reshape((6, 4, 3))


def materialise_vertices_for_composite(
    base: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    cube_w = np.array([1, 0, 0])
    cube_h = np.array([0, 0, 1])
    cube_d = np.array([0, 1, 0])
    width_basis_vector = cube_w
    height_basis_vector = cube_h
    depth_basis_vector = cube_d
    # Shorthand convention is to have the 'bottom-left-front' point as
    # the base, with points defining width/height/depth of the cube
    # after (using the left-hand rule).
    # Note: the ordering of points matters.
    all_cube_points = np.array(
        [
            # bottom-left-front
            base,
            # bottom-left-back
            base + cube_d,
            # bottom-right-back
            base + cube_w + cube_d,
            # bottom-right-front
            base + cube_w,
            # top-left-front
            base + cube_h,
            # top-left-back
            base + cube_h + cube_d,
            # top-right-back
            base + cube_h + cube_w + cube_d,
            # top-right-front
            base + cube_h + cube_w,
        ]
    )

    composite_ranges = list(map(range, map(int, shape)))
    all_cubes_all_points = np.array(
        [
            all_cube_points
            + (w * width_basis_vector)
            + (h * height_basis_vector)
            + (d * depth_basis_vector)
            for (w, h, d) in itertools.product(*composite_ranges)
        ]
    )

    num_cubes = int(math.prod(shape))
    ps = all_cubes_all_points.reshape((num_cubes, 8, 3))

    all_cube_faces = np.array(
        [
            [
                (ps[i][0], ps[i][1], ps[i][2], ps[i][3]),  # bottom
                (ps[i][0], ps[i][4], ps[i][7], ps[i][3]),  # front face
                (ps[i][0], ps[i][1], ps[i][5], ps[i][4]),  # left face
                (ps[i][3], ps[i][7], ps[i][6], ps[i][2]),  # right face
                (ps[i][1], ps[i][5], ps[i][6], ps[i][2]),  # back face
                (ps[i][4], ps[i][5], ps[i][6], ps[i][7]),  # top
            ]
            for i in range(num_cubes)
        ]
    )

    return all_cube_faces.reshape((num_cubes, 6, 4, 3))


class VisualisationBackend:
    def __init__(self, space_transform: np.ndarray) -> None:
        self.figure_not_initialised = True
        # We represent the transform from input points to matplotlib points
        # (assumed in XZY order).
        # TODO: Check this is the correct terminology.
        self.basis = space_transform
        self._collections_per_object = []

    def _initialise_figure_and_ax(self) -> None:
        # Have a function for initialising the figure separately from __init__,
        # as this avoids empty figures appearing in tests unless render() is
        # explicitly called.
        self.fig = plt.figure(figsize=(10, 7))
        self.fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.0, hspace=0.0
        )
        self.ax = self.fig.add_subplot(111, projection="3d")
        # Remove everything except the objects to display.
        self.ax.set_axis_off()
        self.figure_not_initialised = False

    def _convert_basis(self, coordinate: np.ndarray) -> np.ndarray:
        return np.dot(coordinate, self.basis)

    def populate_with_primitive(
        self,
        primitive_id: int,
        base_coordinate: np.ndarray,
        shape: np.ndarray,
        visual_properties: dict[str, Any],
    ) -> None:
        """
        Add the primitive with `primitive_id` with `base_coordinate`, `shape`,
        and `visual_properties` to the output of this backend.

        All input data is assumed to be in space-compatible format, and will be
        transformed to the native format of this backend (e.g. coordinates will
        be mapped to XZY - WHD order).

        # Args
            primitive_id: The ID of the primitive to add.
            base_coordinate: The base coordinate of the primitive.
            shape: The shape to apply, in WHD order.
            visual_properties: Object containing the properties to apply, and
                their values.
        """
        # Currently unused.
        _ = primitive_id

        if self.figure_not_initialised:
            self._initialise_figure_and_ax()

        materialised_vertices = materialise_vertices_for_primitive(
            self._convert_basis(base_coordinate), shape
        )
        self._collections_per_object.append(1)
        assert len(self._collections_per_object) == primitive_id + 1

        matplotlib_like_cube = Poly3DCollection(
            materialised_vertices, **visual_properties
        )
        self.ax.add_collection3d(matplotlib_like_cube)

    def populate_with_composite(
        self,
        composite_id: int,
        base_coordinate: np.ndarray,
        shape: np.ndarray,
        visual_properties: dict[str, Any],
    ) -> None:
        """
        Add the composite with `composite_id` with `base_coordinate`, `shape`,
        and `visual_properties` to the output of this backend.

        All input data is assumed to be in space-compatible format, and will be
        transformed to the native format of this backend (e.g. coordinates will
        be mapped to XZY - WHD order).

        Currently the values of `visual_properties` must be scalars.

        # Args
            composite_id: The IDs of all the primitives to add.
            base_coordinate: The base coordinate of the composite.
            shape: The shape to apply, in WHD order.
            visual_properties: Object containing the properties to apply, and
                their values. Note that the values may be lists, in which case
                each entry corresponds to its respective primitive, or scalars,
                in which case it applies to all primitives.
        """
        shape = shape.flatten()

        if self.figure_not_initialised:
            self._initialise_figure_and_ax()

        materialised_vertices = materialise_vertices_for_composite(
            self._convert_basis(base_coordinate), shape
        )
        volume = math.prod(shape)
        self._collections_per_object.append(volume)
        assert len(self._collections_per_object) == composite_id + 1

        primitive_visual_properties = {
            k: visual_properties[k] for k in visual_properties.keys()
        }

        for primitive_id in range(volume):
            matplotlib_like_cube = Poly3DCollection(
                materialised_vertices[primitive_id],
                **primitive_visual_properties,
            )
            self.ax.add_collection3d(matplotlib_like_cube)

    def mutate_primitive(
        self, primitive_id: int, visual_properties: dict[str, Any]
    ) -> None:
        """
        Mutate the visual properties of the primitive with `primitive_id` in the
        output of this backend, with `visual_properties`.

        # Args
            primitive_id: The ID of the primitive to mutate.
            visual_properties: Object containing the properties to update, and
                their values.
        """
        if self.figure_not_initialised:
            self._initialise_figure_and_ax()

        self.ax.collections[primitive_id].set(**visual_properties)

    def mutate_composite(
        self,
        composite_id: int,
        visual_properties: dict[str, Any],
    ) -> None:
        """
        Mutate the visual properties of the composite with `composite_id` in the
        output of this backend, with `visual_properties`.

        Currently the values of `visual_properties` must be scalars.

        # Args
            composite_id: The ID of the composite to mutate.
            visual_properties: Object containing the properties to update, and
                their values.
        """
        if self.figure_not_initialised:
            self._initialise_figure_and_ax()

        primitive_visual_properties = {
            k: visual_properties[k] for k in visual_properties.keys()
        }

        N = self._collections_per_object[composite_id]
        for primitive_id in range(N):
            offset = N + primitive_id
            self.ax.collections[offset].set(**primitive_visual_properties)

    def transform_primitive(
        self,
        primitive_id: int,
        transform_name: str,
        base_coordinate: np.ndarray,
        shape: np.ndarray,
    ) -> None:
        """
        Transform the primitive with `primitive_id` in the output of this
        backend according to `transform_name`, with `base_coordinate` and
        `shape`.

        # Args
            primitive_id: The ID of the primitive to transform.
            transform_name: The type of transform to apply.
            vertices: The (potentially) updated base coordinate to apply.
            shape: The (potentially) updated shape to apply, in WHD order.
        """
        if self.figure_not_initialised:
            self._initialise_figure_and_ax()

        materialised_vertices = materialise_vertices_for_primitive(
            self._convert_basis(base_coordinate), shape
        )

        if transform_name == "translation":
            _ = shape
            self.ax.collections[primitive_id].set_verts(materialised_vertices)
        if transform_name == "reflection":
            _ = shape
            self.ax.collections[primitive_id].set_verts(materialised_vertices)
        if transform_name == "scale":
            _ = shape
            self.ax.collections[primitive_id].set_verts(materialised_vertices)

    def transform_composite(
        self,
        composite_id: slice,
        transform_name: str,
        base_coordinate: np.ndarray,
        shape: np.ndarray,
    ) -> None:
        """
        Transform the composite with `composite_id` in the output of this
        backend according to `transform_name`, with `base_coordinate` and
        `shape`.

        # Args
            composite_id: The ID of the composite to transform.
            transform_name: The type of transform to apply.
            base_coordinate: The (potentially) updated base coordinate to apply.
            shape: The (potentially) updated shape to apply, in WHD order.
        """
        if self.figure_not_initialised:
            self._initialise_figure_and_ax()

        materialised_vertices = materialise_vertices_for_composite(
            self._convert_basis(base_coordinate), shape
        )

        N = self._collections_per_object[composite_id]
        for primitive_id in range(N):
            offset = N + primitive_id
            if transform_name == "translation":
                _ = shape
                vertices = materialised_vertices[primitive_id]
                self.ax.collections[offset].set_verts(vertices)
            if transform_name == "reflection":
                _ = shape
                vertices = materialised_vertices[primitive_id]
                self.ax.collections[offset].set_verts(vertices)
            if transform_name == "scale":
                raise ValueError(
                    "Scaling for composites is not supported in this backend."
                )
