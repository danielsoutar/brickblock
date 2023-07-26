from dataclasses import dataclass
from typing import Any

import itertools
import numpy as np


@dataclass
class Cube:
    """
    Primitive object for composing scenes.

    This class represents a cube - a six-sided polyhedron with each of its faces
    being squares.

    Internally, coordinates within this Cube are expressed in XZY/WHD format -
    this is due to matplotlib's data layout. However, coordinate parameters
    (e.g. in user-facing functions such as the constructor) are in XYZ/WHD
    format (note the swapping of the axes and dimensions!), and this is the
    preferred format in brickblock.

    # Attributes
        faces: A 6x4x3 array of numbers representing the dense coordinate data
            for this cube. Points are in XZY format.
        facecolor: The color for each of the faces. The default is None, i.e. a
            transparent cube. If this is set, then by default alpha will be 1.
        linewidth: The width for each of the lines.
        edgecolor: The color for each of the lines.
        alpha: The transparency for each of the faces. The default is 0, i.e.
            a transparent cube.
        name: A name for this cube, used for querying within a Space.
    """

    faces: np.ndarray
    facecolor: tuple[float, float, float] | None = None
    linewidth: float = 0.1
    edgecolor: str = "black"
    alpha: float = 0.0
    name: str | None = None

    def __init__(
        self,
        base_vector: np.ndarray,
        scale: float = 1.0,
        facecolor: tuple[float, float, float] | None = None,
        linewidth: float = 0.1,
        edgecolor: str = "black",
        alpha: float | None = None,
        name: str | None = None,
    ) -> None:
        # Users will not expect setting the facecolor only to have the cube be
        # invisible by default, so if the facecolor is set but not the alpha,
        # have the object be fully opaque.
        if alpha is None and facecolor is not None:
            alpha = 1.0

        # On the other hand, the default presentation should be transparent with
        # black lines.
        if alpha is None and facecolor is None:
            alpha = 0.0

        # Check base_vector is 3D.
        is_3d = base_vector.flatten().shape == (3,)
        if not is_3d:
            raise ValueError(
                "Cube objects are three-dimensional, the base vector should be "
                "3D."
            )

        if scale <= 0.0:
            raise ValueError("Cube must have positively-sized dimensions.")

        # Explain this in docs - but essentially this is for navigating around
        # limitation in matplotlib where the Z axis is the vertical one. You
        # cannot just use the camera to fix the problem (I think). Or at least,
        # not with 3D objects and the notion of left/right etc. You need to
        # transpose or flip the actual data (or the axes), and this is the
        # simplest way to achieve this. Of course, the flip-side is that now you
        # are saying the z-axis corresponds to height in Brickblock, which is
        # not ideal.
        # TODO: Have this as a transform for matplotlib and have your own
        # representation instead.
        # The below basis vectors in that order actually define the transform
        # you need. You could reshape the entire matrix in `render` and apply
        # a single matmul to all the data, or just swap the columns.
        w, h, d = base_vector
        base_vector = np.array([w, d, h])
        self._width_basis_vector = np.array([1, 0, 0])
        self._height_basis_vector = np.array([0, 0, 1])
        self._depth_basis_vector = np.array([0, 1, 0])

        points = np.array(
            [
                base_vector,
                scale * self._width_basis_vector,
                scale * self._height_basis_vector,
                scale * self._depth_basis_vector,
            ]
        ).reshape((4, 3))

        full_points = self._construct_points(points)

        self.faces = self._construct_faces(full_points)
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.name = name

    def points(self) -> np.ndarray:
        """
        Get the set of unique points that define this cube.
        """
        return np.array([self.faces[0], self.faces[-1]]).reshape((8, 3))

    def get_visual_metadata(self) -> dict[str, Any]:
        """
        Get the visual properties for this cube.
        """
        return {
            "facecolor": self.facecolor,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "alpha": self.alpha,
        }

    # TODO: Decide on interface - do we want this not-really-bounding box AND
    # points()? What do they mean for Cubes? What do they mean for
    # CompositeCubes? How would this be used by a Space?
    def get_bounding_box(self) -> np.ndarray:
        """
        Get the bounding box around the cube's `points`.

        The output is a 3x2 matrix, with rows in WHD order (xs, zs, ys)
        corresponding to the minimum and maximum per dimension respectively.
        """
        points = np.array([self.faces[0], self.faces[-1]]).reshape((8, 3))
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        z_min = np.min(points[:, 1])
        z_max = np.max(points[:, 1])
        y_min = np.min(points[:, 2])
        y_max = np.max(points[:, 2])

        return np.array(
            [[x_min, x_max], [z_min, z_max], [y_min, y_max]]
        ).reshape((3, 2))

    def _construct_points(self, points: np.ndarray) -> np.ndarray:
        """
        Construct the full set of points from a partial set of points.
        """
        # Shorthand convention is to have the 'bottom-left-front' point as
        # the base, with points defining width/height/depth of the cube
        # after (using the left-hand rule).
        base, w, h, d = points
        # Note: the ordering of points matters.
        full_points = np.array(
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

        return full_points.reshape((8, 3))

    def _construct_faces(self, points: np.ndarray) -> np.ndarray:
        """
        Create the full 6x4x3 coordinate representation of the cube's points.
        """
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


@dataclass
class Cuboid:
    """
    Primitive object for composing scenes.

    Strictly speaking, this defines a 'Rectangular Cuboid' which comprises three
    pairs of rectangles. The more general form can be defined by 8 vertices.

    Internally, coordinates within this Cuboid are expressed in XZY/WHD format -
    this is due to matplotlib's data layout. However, coordinate parameters
    (e.g. in user-facing functions such as the constructor) are in XYZ/WHD
    format (note the swapping of the axes and dimensions!), and this is the
    preferred format in brickblock.

    # Attributes
        faces: A 6x4x3 array of numbers representing the dense coordinate data
            for this cuboid. Points are in XZY format.
        facecolor: The color for each of the faces. The default is None, i.e. a
            transparent cuboid. If this is set, then by default alpha will be 1.
        linewidth: The width for each of the lines.
        edgecolor: The color for each of the lines.
        alpha: The transparency for each of the faces. The default is 0, i.e.
            a transparent cuboid.
        name: A name for this cuboid, used for querying within a Space.
    """

    faces: np.ndarray
    facecolor: tuple[float, float, float] | None = None
    linewidth: float = 0.1
    edgecolor: str = "black"
    alpha: float = 0.0
    name: str | None = None

    # TODO: Decide how to support the simpler and more general cuboids. Maybe
    # rename this to RectangularCuboid?
    def __init__(
        self,
        base_vector: np.ndarray,
        w: float,
        h: float,
        d: float,
        facecolor: tuple[float, float, float] | None = None,
        linewidth: float = 0.1,
        edgecolor: str = "black",
        alpha: float | None = None,
        name: str | None = None,
    ) -> None:
        # Users will not expect setting the facecolor only to have the cube be
        # invisible by default, so if the facecolor is set but not the alpha,
        # have the object be fully opaque.
        if alpha is None and facecolor is not None:
            alpha = 1.0

        # On the other hand, the default presentation should be transparent with
        # black lines.
        if alpha is None and facecolor is None:
            alpha = 0.0

        # Check base_vector is 3D.
        is_3d = base_vector.flatten().shape == (3,)
        if not is_3d:
            raise ValueError(
                "Cuboid objects are three-dimensional, the base vector should "
                "be 3D."
            )

        if w <= 0.0 or h <= 0.0 or d <= 0.0:
            raise ValueError("Cuboid must have positively-sized dimensions.")

        # Explain this in docs - but essentially this is for navigating around
        # limitation in matplotlib where the Z axis is the vertical one. You
        # cannot just use the camera to fix the problem (I think). Or at least,
        # not with 3D objects and the notion of left/right etc. You need to
        # transpose or flip the actual data (or the axes), and this is the
        # simplest way to achieve this. Of course, the flip-side is that now you
        # are saying the z-axis corresponds to height in Brickblock, which is
        # not ideal.
        # TODO: Have this as a transform for matplotlib and have your own
        # representation instead.
        base_w, base_h, base_d = base_vector
        base_vector = np.array([base_w, base_d, base_h])
        self._width_basis_vector = np.array([1, 0, 0])
        self._height_basis_vector = np.array([0, 0, 1])
        self._depth_basis_vector = np.array([0, 1, 0])

        points = np.array(
            [
                base_vector,
                w * self._width_basis_vector,
                h * self._height_basis_vector,
                d * self._depth_basis_vector,
            ]
        ).reshape((4, 3))

        full_points = self._construct_points(points)

        self.faces = self._construct_faces(full_points)
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.name = name

    def points(self) -> np.ndarray:
        """
        Get the set of unique points that define this cuboid.
        """
        return np.array([self.faces[0], self.faces[-1]]).reshape((8, 3))

    def get_visual_metadata(self) -> dict[str, Any]:
        """
        Get the visual properties for this cuboid.
        """
        return {
            "facecolor": self.facecolor,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "alpha": self.alpha,
        }

    # TODO: Decide on interface - do we want this not-really-bounding box AND
    # points()? What do they mean for Cubes/Cuboids/CompositeCubes? How would
    # this be used by a Space?
    def get_bounding_box(self) -> np.ndarray:
        """
        Get the bounding box around the cuboid's `points`.

        The output is a 3x2 matrix, with rows in WHD order (xs, zs, ys)
        corresponding to the minimum and maximum per dimension respectively.
        """
        points = np.array([self.faces[0], self.faces[-1]]).reshape((8, 3))
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        z_min = np.min(points[:, 1])
        z_max = np.max(points[:, 1])
        y_min = np.min(points[:, 2])
        y_max = np.max(points[:, 2])

        return np.array(
            [[x_min, x_max], [z_min, z_max], [y_min, y_max]]
        ).reshape((3, 2))

    def _construct_points(self, points: np.ndarray) -> np.ndarray:
        """
        Construct the full set of points from a partial set of points.
        """
        # Shorthand convention is to have the 'bottom-left-front' point as
        # the base, with points defining width/height/depth of the cube
        # after (using the left-hand rule).
        base, w, h, d = points
        # Note: the ordering of points matters.
        full_points = np.array(
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

        return full_points.reshape((8, 3))

    def _construct_faces(self, points: np.ndarray) -> np.ndarray:
        """
        Create the full 6x4x3 coordinate representation of the cuboid's points.
        """
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


class CompositeCube:
    """
    Composite object for composing scenes.

    Currently this is comprised exclusively of unit cubes - that is, cubes with
    unit scale along each of their dimensions.

    Internally, coordinates within this object are expressed in XZY/WHD format -
    this is due to matplotlib's data layout. However, coordinate parameters
    (e.g. in user-facing functions such as the constructor) are in XYZ/WHD
    format (note the swapping of the axes AND dimensions!), and this is the
    preferred format in brickblock.

    # Attributes
        w: The width of the object, or number of unit-cubes in the width
            dimension.
        h: The height of the object, or number of unit-cubes in the height
            dimension.
        d: The depth of the object, or number of unit-cubes in the depth
            dimension.
        faces: A Nx6x4x3 array of numbers representing the dense coordinate data
            for this object, where N is the product of the three dimensions.
            Points are in XZY format.
        facecolor: The color for each of the faces in every cube. The default is
            None, i.e. transparent cubes. If this is set, then by default alpha
            will be 1.
        linewidth: The width for each of the lines in every cube.
        edgecolor: The color for each of the lines in every cube.
        alpha: The transparency for each of the faces in every cube. The default
            is 0, i.e. transparent cubes.
        style: The visual style of the entire object. Other field values will
            take precedence over this style should they conflict.
        name: A name for this entire object, used for querying within a Space.
    """

    h: int
    w: int
    d: int
    faces: np.ndarray
    facecolor: tuple[float, float, float] | None = None
    linewidth: float = 0.1
    edgecolor: str = "black"
    alpha: float = 0.0
    style: str = "default"
    name: str | None = None

    def __init__(
        self,
        base_vector: np.ndarray,
        w: int,
        h: int,
        d: int,
        facecolor: tuple[float, float, float] | None = None,
        linewidth: float = 0.1,
        edgecolor: str = "black",
        alpha: float | None = None,
        style: str = "default",
        name: str | None = None,
    ) -> None:
        # Users will not expect setting the facecolor only to have the object be
        # invisible by default, so if the facecolor is set but not the alpha,
        # have the object be fully opaque.
        if alpha is None and facecolor is not None:
            alpha = 1.0

        # On the other hand, the default presentation should be transparent with
        # black lines.
        if alpha is None and facecolor is None:
            alpha = 0.0

        # Check base_vector is 3D.
        is_3d = base_vector.flatten().shape == (3,)
        if not is_3d:
            raise ValueError(
                "Composite objects are three-dimensional, the base vector "
                "should be 3D."
            )

        if w <= 0 or h <= 0 or d <= 0:
            raise ValueError(
                "Composite object must have positively-sized dimensions."
            )

        style = style.lower()
        if style not in ["default", "classic"]:
            raise ValueError("Composite object was given an invalid style.")

        # Explain this in docs - but essentially this is for navigating around
        # limitation in matplotlib where the Z axis is the vertical one. You
        # cannot just use the camera to fix the problem (I think). Or at least,
        # not with 3D objects and the notion of left/right etc. You need to
        # transpose or flip the actual data (or the axes), and this is the
        # simplest way to achieve this. Of course, the flip-side is that now you
        # are saying the z-axis corresponds to height in Brickblock, which is
        # not ideal.
        # TODO: Have this as a transform for matplotlib and have your own
        # representation instead.
        base_w, base_h, base_d = base_vector
        base_vector = np.array([base_w, base_d, base_h])
        self._width_basis_vector = np.array([1, 0, 0])
        self._height_basis_vector = np.array([0, 0, 1])
        self._depth_basis_vector = np.array([0, 1, 0])

        # For now we assume that composites are built out of unit cubes.
        # This could be generalised to arbitrary cubes but for now this will do.
        points = np.array(
            [
                base_vector,
                self._width_basis_vector,
                self._height_basis_vector,
                self._depth_basis_vector,
            ]
        ).reshape((4, 3))

        full_points = self._construct_points(points, w, h, d)

        self.w = w
        self.h = h
        self.d = d
        self.faces = self._construct_faces(full_points)
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.style = style
        self.name = name

    def points(self) -> np.ndarray:
        """
        Get the set of unique points that define this object.
        """
        # TODO: Figure out the relevant points that define the bounds of the
        # entire object.
        return np.array([]).reshape(())

    def get_visual_metadata(self) -> dict[str, Any]:
        """
        Get the visual properties for this object.
        """
        return {
            "facecolor": self.facecolor,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "alpha": self.alpha,
        }

    # TODO: Decide on interface - do we want this not-really-bounding box AND
    # points()? What do they mean for Cubes? What do they mean for
    # CompositeCubes? How would this be used by a Space?
    def get_bounding_box(self) -> np.ndarray:
        """
        TODO: See above.
        """
        return np.array().reshape((3, 2))

    def _construct_points(
        self,
        cube_points: np.ndarray,
        composite_w: float,
        composite_h: float,
        composite_d: float,
    ) -> np.ndarray:
        """
        Construct the full set of points from a partial set of points.
        """
        # Shorthand convention is to have the 'bottom-left-front' point as
        # the base, with points defining width/height/depth of the cube
        # after (using the left-hand rule).
        base, cube_w, cube_h, cube_d = cube_points
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

        all_cube_points = all_cube_points.reshape((8, 3))

        all_cubes_all_points = np.array(
            [
                all_cube_points
                + (w * self._width_basis_vector)
                + (h * self._height_basis_vector)
                + (d * self._depth_basis_vector)
                for (w, h, d) in itertools.product(
                    range(composite_w), range(composite_h), range(composite_d)
                )
            ]
        )

        return all_cubes_all_points.reshape(
            (composite_w, composite_h, composite_d, 8, 3)
        )

    def _construct_faces(self, points: np.ndarray) -> np.ndarray:
        """
        Create the full 6x4x3 coordinate representation of the object's points.
        """
        w, h, d, cube_points, num_coords = points.shape
        num_cubes = w * h * d
        ps = points.reshape((num_cubes, cube_points, num_coords))

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
