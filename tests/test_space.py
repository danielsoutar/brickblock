import pytest

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import brickblock as bb


def mock_coordinates_entry() -> np.ndarray:
    base = np.array(
        [[[0., 0., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 0., 0.]],
       [[0., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [1., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 1.],
        [0., 1., 0.]],
       [[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.],
        [1., 0., 1.]],
       [[0., 0., 1.],
        [0., 1., 1.],
        [1., 1., 1.],
        [1., 0., 1.]],
       [[0., 1., 0.],
        [0., 1., 1.],
        [1., 1., 1.],
        [1., 1., 0.]]]
    ).reshape((1, 6, 4, 3))

    return base


def test_space_creation() -> None:
    space = bb.Space()

    assert np.array_equal(space.dims, np.zeros((3, 2)))
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


def test_space_snapshot_creates_a_scene() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    space.add_cube(cube)
    space.snapshot()

    assert np.array_equal(space.dims, np.array([[0, 1], [0, 1], [0, 1]]))
    assert np.array_equal(space.mean, np.array([[0.5], [0.5], [0.5]]))
    assert np.array_equal(space.total, np.array([[0.5], [0.5], [0.5]]))
    assert space.num_objs == 1
    assert space.primitive_counter == 1
    assert space.time_step == 1
    assert space.scene_counter == 1
    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry(),
                np.zeros((expected_num_entries-1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        'facecolor': [None],
        'linewidths': [0.1],
        'edgecolor': ['black'],
        'alpha': [0.0],
    }
    assert space.cuboid_index == {0: {0: [0]}}
    assert space.changelog == [bb.Addition(timestep_id=0, name=None)]


def test_space_multiple_snapshots_create_multiple_scenes() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    space.add_cube(cube)
    space.snapshot()

    other_points = np.array(
        [(3, 3, 3), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    second_cube = bb.Cube(other_points)
    space.add_cube(second_cube)
    space.snapshot()

    assert np.array_equal(space.dims, np.array([[0, 4], [0, 4], [0, 4]]))
    assert np.array_equal(space.mean, np.array([[2.0], [2.0], [2.0]]))
    assert np.array_equal(space.total, np.array([[4.0], [4.0], [4.0]]))
    assert space.num_objs == 2
    assert space.primitive_counter == 2
    assert space.time_step == 2
    assert space.scene_counter == 2
    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry(),
                mock_coordinates_entry() + 3,
                np.zeros((expected_num_entries-2, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        'facecolor': [None, None],
        'linewidths': [0.1, 0.1],
        'edgecolor': ['black', 'black'],
        'alpha': [0.0, 0.0],
    }
    assert space.cuboid_index == {0: {0: [0]}, 1: {1: [1]}}
    assert space.changelog == [
        bb.Addition(timestep_id=0, name=None),
        bb.Addition(timestep_id=1, name=None),
    ]


def test_space_creates_distinct_scenes_only() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    space.add_cube(cube)
    space.snapshot()

    expected_err_msg = (
        "A snapshot must include at least one addition, mutation, or deletion "
        "in the given scene."
    )
    with pytest.raises(Exception, match=expected_err_msg):
        space.snapshot()


def test_space_creates_valid_axes_on_render() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    space.add_cube(cube)
    space.snapshot()
    _, ax = space.render()

    plt_internal_data = ax.collections[0]._vec
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cube.faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_space_creates_valid_axes_on_render_multiple_scenes() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    space.add_cube(cube)
    space.snapshot()
    # Check this runs without issues, but we don't need the fig for this test.
    space.render()

    other_points = np.array(
        [(3, 3, 3), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    second_cube = bb.Cube(other_points)
    space.add_cube(second_cube)
    space.snapshot()
    _, ax2 = space.render()

    plt_internal_data_for_first_cube = ax2.collections[0]._vec.T
    plt_internal_data_for_second_cube = ax2.collections[1]._vec.T
    plt_internal_reshaped_data = np.concatenate(
        [plt_internal_data_for_first_cube, plt_internal_data_for_second_cube],
        axis=0
    ).reshape((2, 6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_first_cube = np.concatenate([cube.faces, ones], -1)
    original_augmented_second_cube = np.concatenate([second_cube.faces, ones], -1)

    expected_data = np.stack(
        [original_augmented_first_cube, original_augmented_second_cube],
        axis=0
    )

    assert np.array_equal(expected_data, plt_internal_reshaped_data)


def test_space_add_multiple_cubes_in_single_scene() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    other_points = np.array(
        [(3, 3, 3), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    second_cube = bb.Cube(other_points)

    space.add_cube(cube)
    space.add_cube(second_cube)
    space.snapshot()

    assert space.num_objs == 2
    assert space.primitive_counter == 2
    assert space.time_step == 2
    assert space.scene_counter == 1

    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry(),
                mock_coordinates_entry() + 3,
                np.zeros((expected_num_entries-2, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        'facecolor': [None, None],
        'linewidths': [0.1, 0.1],
        'edgecolor': ['black', 'black'],
        'alpha': [0.0, 0.0],
    }
    assert space.cuboid_index == {0: {0: [0], 1: [1]}}
    assert space.changelog == [
        bb.Addition(timestep_id=0, name=None),
        bb.Addition(timestep_id=1, name=None),
    ]


def test_space_creates_valid_axes_on_render_multiple_cubes_single_scene() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    other_points = np.array(
        [(3, 3, 3), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    second_cube = bb.Cube(other_points)

    space.add_cube(cube)
    space.add_cube(second_cube)
    space.snapshot()
    _, ax = space.render()

    plt_internal_data_for_first_cube = ax.collections[0]._vec.T
    plt_internal_data_for_second_cube = ax.collections[1]._vec.T
    plt_internal_reshaped_data = np.concatenate(
        [plt_internal_data_for_first_cube, plt_internal_data_for_second_cube],
        axis=0
    ).reshape((2, 6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_first_cube = np.concatenate([cube.faces, ones], -1)
    original_augmented_second_cube = np.concatenate([second_cube.faces, ones], -1)

    expected_data = np.stack(
        [original_augmented_first_cube, original_augmented_second_cube],
        axis=0
    )

    assert np.array_equal(expected_data, plt_internal_reshaped_data)


def test_space_creates_valid_axes_on_render_multiple_cubes_scenes() -> None:
    space = bb.Space()

    points = np.array(
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    cube = bb.Cube(points)
    other_points = np.array(
        [(3, 3, 3), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    second_cube = bb.Cube(other_points)
    more_points = np.array(
        [(1, 1, 1), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ).reshape((4, 3))
    third_cube = bb.Cube(more_points)
    yet_more_points = np.array(
        [(2, 2, 2), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    )
    fourth_cube = bb.Cube(yet_more_points)

    space.add_cube(cube)
    space.add_cube(second_cube)
    space.snapshot()
    # Check this runs without issues, but we don't need the fig for this test.
    space.render()
    space.add_cube(third_cube)
    space.add_cube(fourth_cube)

    space.snapshot()
    fig, ax = space.render()

    plt_internal_data_for_cubes = [ax.collections[i]._vec.T for i in range(4)]
    plt_internal_reshaped_data = np.concatenate(
        plt_internal_data_for_cubes,
        axis=0
    ).reshape((4, 6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_cubes = [
        np.concatenate([c.faces, ones], -1)
        for c in [cube, second_cube, third_cube, fourth_cube]
    ]

    expected_data = np.stack(original_augmented_cubes, axis=0)

    assert np.array_equal(expected_data, plt_internal_reshaped_data)
