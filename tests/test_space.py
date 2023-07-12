import pytest

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import brickblock as bb


def test_space_creation() -> None:
    space = bb.Space()
    assert space.dims is None
    assert np.array_equal(space.mean, np.zeros((3, 1)))
    assert np.array_equal(space.total, np.zeros((3, 1)))
    assert space.num_objs == 0
    assert space.primitive_counter == 0
    assert space.time_step == 0
    assert space.scene_counter == 0
    assert np.array_equal(space.cuboid_coordinates, np.zeros((10, 6, 4, 3)))
    assert space.cuboid_visual_metadata == {}
    assert space.cuboid_index is not None
    assert space.changelog == []


def test_space_creates_valid_axes_on_render() -> None:
    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)

    space = bb.Space()
    space.add_cube(cube)
    fig, ax = space.render()
    plt_internal_data = np.array([ax.collections[0]._vec])

    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cube.faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)
