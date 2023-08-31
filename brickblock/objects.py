from dataclasses import dataclass
from typing import Any

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
        w: The width of the object.
        h: The height of the object.
        d: The depth of the object.
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

    w: float
    h: float
    d: float
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
        base_vector = base_vector.flatten()
        is_3d = base_vector.shape == (3,)
        if not is_3d:
            raise ValueError(
                "Cube objects are three-dimensional, the base vector should be "
                "3D."
            )

        if scale <= 0.0:
            raise ValueError("Cube must have positively-sized dimensions.")

        self.base = base_vector
        self.w = scale
        self.h = scale
        self.d = scale
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.name = name

    def shape(self) -> tuple[float, float, float]:
        return (self.w, self.h, self.d)

    def visual_metadata(self) -> dict[str, Any]:
        """
        Get the visual properties for this cube.
        """
        return {
            "facecolor": self.facecolor,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "alpha": self.alpha,
        }


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
        w: The width of the object.
        h: The height of the object.
        d: The depth of the object.
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

    w: float
    h: float
    d: float
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
        base_vector = base_vector.flatten()
        is_3d = base_vector.shape == (3,)
        if not is_3d:
            raise ValueError(
                "Cuboid objects are three-dimensional, the base vector should "
                "be 3D."
            )

        if w <= 0.0 or h <= 0.0 or d <= 0.0:
            raise ValueError("Cuboid must have positively-sized dimensions.")

        self.base = base_vector
        self.w = w
        self.h = h
        self.d = d
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.name = name

    def shape(self) -> tuple[float, float, float]:
        return (self.w, self.h, self.d)

    def visual_metadata(self) -> dict[str, Any]:
        """
        Get the visual properties for this cuboid.
        """
        return {
            "facecolor": self.facecolor,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "alpha": self.alpha,
        }


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

    w: int
    h: int
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
        base_vector = base_vector.flatten()
        is_3d = base_vector.shape == (3,)
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

        self.base = base_vector
        self.w = w
        self.h = h
        self.d = d
        self.facecolor = facecolor
        self.linewidth = linewidth
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.style = style
        self.name = name

    def shape(self) -> tuple[float, float, float]:
        return (float(self.w), float(self.h), float(self.d))

    def visual_metadata(self) -> dict[str, Any]:
        """
        Get the visual properties for this object.
        """
        return {
            "facecolor": self.facecolor,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "alpha": self.alpha,
        }
