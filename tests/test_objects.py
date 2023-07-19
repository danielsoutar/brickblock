import pytest

import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import brickblock as bb


def test_cube_creation() -> None:
    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)

    assert cube.faces.shape == (6, 4, 3)
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


def test_cube_creates_all_data_needed_for_visualising() -> None:
    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    poly = Poly3DCollection(cube.faces)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt_collection = ax.add_collection3d(poly).axes.collections[0]
    # The internal vector includes all 1s in an the implicit 4th dimension
    # TODO: Understand why this is necessary. Probably to do with 3D
    # projections or something like that.
    plt_internal_data = np.array([plt_collection._vec])
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cube.faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_cube_visualisation_can_be_customised() -> None:
    red, green, blue = 1.0, 0.1569, 0.0
    alpha = 0.1
    linewidth = 0.5

    cube = bb.Cube(
        base_vector=np.array([0, 0, 0]),
        scale=3.0,
        facecolor=(red, green, blue),
        linewidth=linewidth,
        alpha=alpha,
    )
    poly = Poly3DCollection(cube.faces)
    poly.set_facecolor(cube.facecolor)
    poly.set_alpha(cube.alpha)
    poly.set_linewidth(cube.linewidth)
    poly.set_edgecolor(cube.edgecolor)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt_collection = ax.add_collection3d(poly).axes.collections[0]

    # Check colours
    expected_rgba = np.array([red, green, blue, alpha]).reshape((1, 4))
    actual_rgba = plt_collection._facecolor3d
    assert np.array_equal(expected_rgba, actual_rgba)

    # Check lines
    expected_linewidths = np.array([linewidth])
    actual_linewidths = plt_collection._linewidths
    assert np.array_equal(expected_linewidths, actual_linewidths)
    expected_edgecolors = np.array([0.0, 0.0, 0.0, alpha]).reshape((1, 4))
    actual_edgecolors = plt_collection._edgecolors
    assert np.array_equal(expected_edgecolors, actual_edgecolors)


def test_composite_cube_creation() -> None:
    h, w, d = 3, 4, 3
    composite = bb.CompositeCube(base_vector=np.array([0, 0, 0]), h=h, w=w, d=d)

    num_cubes = h * w * d

    assert composite.h == h
    assert composite.w == w
    assert composite.d == d
    assert composite.faces.shape == (num_cubes, 6, 4, 3)
    assert composite.facecolor is None
    assert composite.linewidth == 0.1
    assert composite.edgecolor == "black"
    assert composite.alpha == 0.0
    assert composite.name is None


def test_invalid_dims_throws_exception_making_composite_cube() -> None:
    invalid_dims = {"h": 2, "w": -1, "d": 4}

    expected_err_msg = "Composite cube must have positively-sized dimensions."

    with pytest.raises(ValueError, match=expected_err_msg):
        bb.CompositeCube(base_vector=np.array([0, 0, 0]), **invalid_dims)


def test_all_cubes_in_composite_cube_have_same_dims() -> None:
    h, w, d = 3, 4, 2
    composite = bb.CompositeCube(base_vector=np.array([0, 0, 0]), h=h, w=w, d=d)

    faces_per_cube = composite.faces

    first_cube = faces_per_cube[0]

    # If each cube is merely an offset version of the first, then they must have
    # equal dimensions.
    height_basis_vector = np.array([0, 0, 1])
    width_basis_vector = np.array([1, 0, 0])
    depth_basis_vector = np.array([0, 1, 0])

    for i in range(h):
        for j in range(w):
            for k in range(d):
                idx = (i * w * d) + (j * d) + k
                current_cube = faces_per_cube[idx]
                current_cube_offset = (
                    current_cube
                    - (i * height_basis_vector)
                    - (j * width_basis_vector)
                    - (k * depth_basis_vector)
                )
                assert np.array_equal(first_cube, current_cube_offset)


def test_composite_cube_creates_all_data_needed_for_visualising() -> None:
    h, w, d = 3, 4, 2
    composite = bb.CompositeCube(base_vector=np.array([0, 0, 0]), h=h, w=w, d=d)

    num_cubes = h * w * d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ones = np.ones((6, 4, 1))

    for i in range(num_cubes):
        poly = Poly3DCollection(composite.faces[i])
        plt_collection = ax.add_collection3d(poly).axes.collections[i]
        plt_internal_data = np.array([plt_collection._vec])
        plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

        original_augmented_data = np.concatenate([composite.faces[i], ones], -1)

        assert np.array_equal(
            plt_internal_reshaped_data, original_augmented_data
        )


def test_cuboid_creation() -> None:
    cuboid = bb.Cuboid(base_vector=np.array([0, 0, 0]), h=2.0, w=4.0, d=6.0)

    assert cuboid.faces.shape == (6, 4, 3)
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


def test_cuboid_creates_all_data_needed_for_visualising() -> None:
    cuboid = bb.Cuboid(base_vector=np.array([0, 0, 0]), h=2.0, w=4.0, d=6.0)
    poly = Poly3DCollection(cuboid.faces)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt_collection = ax.add_collection3d(poly).axes.collections[0]
    # The internal vector includes all 1s in an the implicit 4th dimension
    # TODO: Understand why this is necessary. Probably to do with 3D
    # projections or something like that.
    plt_internal_data = np.array([plt_collection._vec])
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cuboid.faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_objects_can_have_names() -> None:
    cuboid = bb.Cuboid(
        base_vector=np.array([0, 0, 0]),
        h=2.0,
        w=4.0,
        d=6.0,
        name="my-first-cuboid",
    )

    assert cuboid.name == "my-first-cuboid"
