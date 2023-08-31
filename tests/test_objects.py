import pytest

import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import brickblock as bb


def test_cube_creation() -> None:
    cube = bb.Cube(base_vector=np.array([0, 0, 0]))

    assert np.array_equal(cube.base, np.array([0, 0, 0]))
    assert cube.facecolor is None
    assert cube.linewidth == 0.1
    assert cube.edgecolor == "black"
    assert cube.alpha == 0.0
    assert cube.name is None


def test_invalid_base_throws_exception_making_cube() -> None:
    invalid_base_vector = np.array([0.0, 0.0])

    expected_err_msg = (
        "Cube objects are three-dimensional, the base vector should be 3D."
    )
    with pytest.raises(ValueError, match=expected_err_msg):
        bb.Cube(invalid_base_vector, scale=1.0)


def test_invalid_scale_throws_exception_making_cube() -> None:
    expected_err_msg = "Cube must have positively-sized dimensions."
    with pytest.raises(ValueError, match=expected_err_msg):
        bb.Cube(base_vector=np.array([0, 0, 0]), scale=0.0)
    with pytest.raises(ValueError, match=expected_err_msg):
        bb.Cube(base_vector=np.array([0, 0, 0]), scale=-1.0)


def test_composite_cube_creation() -> None:
    w, h, d = 4, 3, 3
    composite = bb.CompositeCube(base_vector=np.array([0, 0, 0]), w=w, h=h, d=d)

    assert np.array_equal(composite.base, np.array([0, 0, 0]))
    assert composite.w == w
    assert composite.h == h
    assert composite.d == d
    assert composite.facecolor is None
    assert composite.linewidth == 0.1
    assert composite.edgecolor == "black"
    assert composite.alpha == 0.0
    assert composite.style == "default"
    assert composite.name is None


def test_invalid_dims_throws_exception_making_composite_cube() -> None:
    invalid_dims = {"h": 2, "w": -1, "d": 4}

    expected_err_msg = "Composite object must have positively-sized dimensions."

    with pytest.raises(ValueError, match=expected_err_msg):
        bb.CompositeCube(base_vector=np.array([0, 0, 0]), **invalid_dims)


def test_cuboid_creation() -> None:
    cuboid = bb.Cuboid(base_vector=np.array([0, 0, 0]), w=4.0, h=2.0, d=6.0)

    assert np.array_equal(cuboid.base, np.array([0, 0, 0]))
    assert cuboid.facecolor is None
    assert cuboid.linewidth == 0.1
    assert cuboid.edgecolor == "black"
    assert cuboid.alpha == 0.0
    assert cuboid.name is None


def test_invalid_dims_throws_exception_making_cuboid() -> None:
    invalid_dims = {"h": 2, "w": -1, "d": 4}

    expected_err_msg = "Cuboid must have positively-sized dimensions."

    with pytest.raises(ValueError, match=expected_err_msg):
        bb.Cuboid(base_vector=np.array([0, 0, 0]), **invalid_dims)


def test_objects_can_have_names() -> None:
    cuboid = bb.Cuboid(
        base_vector=np.array([0, 0, 0]),
        w=4.0,
        h=2.0,
        d=6.0,
        name="my-first-cuboid",
    )

    assert cuboid.name == "my-first-cuboid"


def test_composite_can_have_classic_style() -> None:
    composite = bb.CompositeCube(
        base_vector=np.array([0, 0, 0]), w=4, h=2, d=6, style="classic"
    )

    assert composite.style == "classic"


def test_composite_cannot_have_invalid_style() -> None:
    with pytest.raises(
        ValueError, match="Composite object was given an invalid style."
    ):
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]), w=4, h=2, d=6, style="some-style"
        )
