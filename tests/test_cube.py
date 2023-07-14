import pytest

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import brickblock as bb


def test_cube_creation() -> None:
    test_cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)

    assert test_cube.faces.shape == (6, 4, 3)
    assert test_cube.facecolor == None
    assert test_cube.linewidth == 0.1
    assert test_cube.edgecolor == 'black'
    assert test_cube.alpha == 0.0


def test_invalid_base_throws_exception_making_cube() -> None:
    invalid_base_vector = np.array([0.0, 0.0])

    expected_err_msg = (
        "Cube objects are three-dimensional, the base vector should be 3D."
    )
    with pytest.raises(ValueError, match=expected_err_msg):
        bb.Cube(invalid_base_vector, scale=1.0)


def test_cube_creates_all_data_needed_for_visualising() -> None:
    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    poly = Poly3DCollection(cube.faces)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt_collection = ax.add_collection3d(poly).axes.collections[0]
    # The internal vector includes all 1s in an the implicit 4th dimension
    # TODO: Understand why this is necessary. Probably to do with 3D projections
    # or something like that.
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
    ax = fig.add_subplot(111, projection='3d')
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