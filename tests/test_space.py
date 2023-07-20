import itertools
import pytest

import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import brickblock as bb


def mock_coordinates_entry() -> np.ndarray:
    # Points here are in XZY order
    point0 = np.array([0.0, 0.0, 0.0])
    point1 = np.array([0.0, 1.0, 0.0])
    point2 = np.array([1.0, 1.0, 0.0])
    point3 = np.array([1.0, 0.0, 0.0])
    point4 = np.array([0.0, 0.0, 1.0])
    point5 = np.array([0.0, 1.0, 1.0])
    point6 = np.array([1.0, 1.0, 1.0])
    point7 = np.array([1.0, 0.0, 1.0])

    base = np.array(
        [
            [point0, point1, point2, point3],
            [point0, point4, point7, point3],
            [point0, point1, point5, point4],
            [point3, point7, point6, point2],
            [point1, point5, point6, point2],
            [point4, point5, point6, point7],
        ]
    ).reshape((1, 6, 4, 3))

    return base


def mock_cuboid_coordinates_entry() -> np.ndarray:
    # Points here are in XZY order
    point0 = np.array([0.0, 0.0, 0.0])
    point1 = np.array([0.0, 6.0, 0.0])
    point2 = np.array([4.0, 6.0, 0.0])
    point3 = np.array([4.0, 0.0, 0.0])
    point4 = np.array([0.0, 0.0, 2.0])
    point5 = np.array([0.0, 6.0, 2.0])
    point6 = np.array([4.0, 6.0, 2.0])
    point7 = np.array([4.0, 0.0, 2.0])

    base = np.array(
        [
            [point0, point1, point2, point3],
            [point0, point4, point7, point3],
            [point0, point1, point5, point4],
            [point3, point7, point6, point2],
            [point1, point5, point6, point2],
            [point4, point5, point6, point7],
        ]
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

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
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
                np.zeros((expected_num_entries - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None],
        "linewidth": [0.1],
        "edgecolor": ["black"],
        "alpha": [0.0],
    }
    assert space.cuboid_index == {0: {0: [0]}}
    assert space.changelog == [bb.Addition(timestep_id=0, name=None)]


def test_space_multiple_snapshots_create_multiple_scenes() -> None:
    space = bb.Space()

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    space.add_cube(cube)
    space.snapshot()

    second_cube = bb.Cube(base_vector=np.array([3, 3, 3]), scale=1.0)
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
                np.zeros((expected_num_entries - 2, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None, None],
        "linewidth": [0.1, 0.1],
        "edgecolor": ["black", "black"],
        "alpha": [0.0, 0.0],
    }
    assert space.cuboid_index == {0: {0: [0]}, 1: {1: [1]}}
    assert space.changelog == [
        bb.Addition(timestep_id=0, name=None),
        bb.Addition(timestep_id=1, name=None),
    ]


def test_space_creates_distinct_scenes_only() -> None:
    space = bb.Space()

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
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

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    space.add_cube(cube)
    space.snapshot()
    _, ax = space.render()

    plt_internal_data = ax.collections[0]._vec
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cube.faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_space_does_nothing_on_render_when_empty() -> None:
    ...


def test_space_creates_valid_axes_on_render_multiple_scenes() -> None:
    space = bb.Space()

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    space.add_cube(cube)
    space.snapshot()
    # Check this runs without issues, but we don't need the fig for this test.
    space.render()

    second_cube = bb.Cube(base_vector=np.array([3, 3, 3]), scale=1.0)
    space.add_cube(second_cube)
    space.snapshot()
    _, ax2 = space.render()

    plt_internal_data_for_first_cube = ax2.collections[0]._vec.T
    plt_internal_data_for_second_cube = ax2.collections[1]._vec.T
    plt_internal_reshaped_data = np.concatenate(
        [plt_internal_data_for_first_cube, plt_internal_data_for_second_cube],
        axis=0,
    ).reshape((2, 6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_first_cube = np.concatenate([cube.faces, ones], -1)
    original_augmented_second_cube = np.concatenate(
        [second_cube.faces, ones], -1
    )

    expected_data = np.stack(
        [original_augmented_first_cube, original_augmented_second_cube], axis=0
    )

    assert np.array_equal(expected_data, plt_internal_reshaped_data)


def test_space_add_multiple_cubes_in_single_scene() -> None:
    space = bb.Space()

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    second_cube = bb.Cube(base_vector=np.array([3, 3, 3]), scale=1.0)

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
                np.zeros((expected_num_entries - 2, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None, None],
        "linewidth": [0.1, 0.1],
        "edgecolor": ["black", "black"],
        "alpha": [0.0, 0.0],
    }
    assert space.cuboid_index == {0: {0: [0], 1: [1]}}
    assert space.changelog == [
        bb.Addition(timestep_id=0, name=None),
        bb.Addition(timestep_id=1, name=None),
    ]


def test_space_creates_valid_axes_on_render_multiple_cubes_single_scene() -> (
    None
):
    space = bb.Space()

    cube = bb.Cube(base_vector=np.array([0, 0, 0]), scale=1.0)
    second_cube = bb.Cube(base_vector=np.array([3, 2, 1]), scale=1.0)

    space.add_cube(cube)
    space.add_cube(second_cube)
    space.snapshot()
    _, ax = space.render()

    plt_internal_data_for_first_cube = ax.collections[0]._vec.T
    plt_internal_data_for_second_cube = ax.collections[1]._vec.T
    plt_internal_reshaped_data = np.concatenate(
        [plt_internal_data_for_first_cube, plt_internal_data_for_second_cube],
        axis=0,
    ).reshape((2, 6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_first_cube = np.concatenate([cube.faces, ones], -1)
    original_augmented_second_cube = np.concatenate(
        [second_cube.faces, ones], -1
    )

    expected_data = np.stack(
        [original_augmented_first_cube, original_augmented_second_cube], axis=0
    )

    assert np.array_equal(expected_data, plt_internal_reshaped_data)


def test_space_creates_valid_axes_on_render_multiple_cubes_scenes() -> None:
    space = bb.Space()

    cube = bb.Cube(base_vector=np.array([0, 0, 0]))
    second_cube = bb.Cube(base_vector=np.array([7, 8, 9]), facecolor="black")
    third_cube = bb.Cube(base_vector=np.array([1, 2, 3]), facecolor="blue")
    fourth_cube = bb.Cube(base_vector=np.array([4, 5, 6]), facecolor="red")

    space.add_cube(cube)
    space.add_cube(second_cube)
    space.snapshot()
    # Check this runs without issues, but we don't need the fig for this test.
    space.render()
    space.add_cube(third_cube)
    space.add_cube(fourth_cube)

    space.snapshot()
    _, ax = space.render()

    plt_internal_data_for_cubes = [ax.collections[i]._vec.T for i in range(4)]
    plt_internal_reshaped_data = np.concatenate(
        plt_internal_data_for_cubes, axis=0
    ).reshape((4, 6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_cubes = [
        np.concatenate([c.faces, ones], -1)
        for c in [cube, second_cube, third_cube, fourth_cube]
    ]

    expected_data = np.stack(original_augmented_cubes, axis=0)

    assert np.array_equal(expected_data, plt_internal_reshaped_data)


def test_space_can_customise_cube_visual_properties() -> None:
    space = bb.Space()

    red, green, blue = 1.0, 0.1569, 0.0
    alpha = 0.1
    linewidth = 0.5

    cube = bb.Cube(
        base_vector=np.array([0, 0, 0]),
        facecolor=(red, green, blue),
        linewidth=linewidth,
        alpha=alpha,
    )
    space.add_cube(cube)
    assert space.cuboid_visual_metadata == {
        "facecolor": [(red, green, blue)],
        "linewidth": [linewidth],
        "edgecolor": ["black"],
        "alpha": [alpha],
    }

    _, ax = space.render()

    plt_collection = ax.axes.collections[0]

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


def test_space_can_add_composite_cube() -> None:
    space = bb.Space()

    w, h, d = 4, 3, 2
    composite = bb.CompositeCube(base_vector=np.array([0, 0, 0]), w=w, h=h, d=d)

    num_cubes = w * h * d

    space.add_composite(composite)
    space.snapshot()

    assert np.array_equal(space.dims, np.array([[0, 4], [0, 2], [0, 3]]))
    assert np.array_equal(space.mean, np.array([[2.0], [1.0], [1.5]]))
    assert np.array_equal(space.total, np.array([[48.0], [24.0], [36.0]]))
    assert space.num_objs == 1
    assert space.primitive_counter == num_cubes
    assert space.time_step == 1
    assert space.scene_counter == 1

    # The initial number of entries is 10, and the array size is doubled on
    # overflow. Hence we'd expect re-allocating 40 entries when overflowing 20.
    expected_num_entries = 40
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                *[
                    mock_coordinates_entry()
                    + (w * width)
                    + (h * height)
                    + (d * depth)
                    for (w, h, d) in itertools.product(
                        range(w), range(h), range(d)
                    )
                ],
                np.zeros((expected_num_entries - num_cubes, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None] * num_cubes,
        "linewidth": [0.1] * num_cubes,
        "edgecolor": ["black"] * num_cubes,
        "alpha": [0.0] * num_cubes,
    }
    assert space.cuboid_index == {0: {0: [i for i in range(num_cubes)]}}
    assert space.changelog == [bb.Addition(timestep_id=0, name=None)]


def test_space_creates_valid_axes_on_render_for_composite() -> None:
    space = bb.Space()

    w, h, d = 4, 3, 2
    num_cubes = w * h * d

    composite = bb.CompositeCube(
        base_vector=np.array([0, 0, 0]),
        w=w,
        h=h,
        d=d,
        facecolor="red",
    )
    second_composite = bb.CompositeCube(
        base_vector=np.array([w, h, d]),
        w=w,
        h=h,
        d=d,
        facecolor="green",
    )
    space.add_composite(composite)
    space.add_composite(second_composite)
    _, ax = space.render()

    for i in range(num_cubes):
        plt_internal_data = ax.collections[i]._vec
        plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

        # Add the implicit 4th dimension to the original data - all ones.
        ones = np.ones((6, 4, 1))
        original_augmented_data = np.concatenate([composite.faces[i], ones], -1)

        assert np.array_equal(
            original_augmented_data, plt_internal_reshaped_data
        )


def test_space_can_add_cuboid() -> None:
    space = bb.Space()

    w, h, d = 4, 2, 6
    cuboid = bb.Cuboid(base_vector=np.array([0, 0, 0]), w=w, h=h, d=d)

    space.add_cuboid(cuboid)
    space.snapshot()

    assert np.array_equal(space.dims, np.array([[0, 4], [0, 6], [0, 2]]))
    assert np.array_equal(space.mean, np.array([[2], [3], [1]]))
    assert np.array_equal(space.total, np.array([[2], [3], [1]]))
    assert space.num_objs == 1
    assert space.primitive_counter == 1
    assert space.time_step == 1
    assert space.scene_counter == 1
    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_cuboid_coordinates_entry(),
                np.zeros((expected_num_entries - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None],
        "linewidth": [0.1],
        "edgecolor": ["black"],
        "alpha": [0.0],
    }
    assert space.cuboid_index == {0: {0: [0]}}
    assert space.changelog == [bb.Addition(timestep_id=0, name=None)]


def test_space_creates_valid_axes_on_render_for_cuboid() -> None:
    space = bb.Space()

    cuboid = bb.Cuboid(base_vector=np.array([0, 0, 0]), w=4.0, h=2.0, d=6.0)
    space.add_cuboid(cuboid)
    space.snapshot()
    _, ax = space.render()

    plt_internal_data = ax.collections[0]._vec
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cuboid.faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_space_can_add_named_objects() -> None:
    space = bb.Space()

    w, h, d = 4, 4, 3
    num_cubes_in_input_tensor = w * h * d
    f_w, f_h, f_d = 2, 2, 3
    num_cubes_in_filter_tensor = f_w * f_h * f_d

    input_tensor = bb.CompositeCube(
        base_vector=np.array([0, 0, 0]),
        w=w,
        h=h,
        d=d,
        name="input-tensor",
    )
    filter_tensor = bb.CompositeCube(
        base_vector=np.array([0, 2, 0]),
        w=f_w,
        h=f_h,
        d=f_d,
        name="filter-tensor",
    )
    space.add_composite(input_tensor)
    space.add_composite(filter_tensor)
    space.snapshot()
    _, ax = space.render()

    assert space.cuboid_names == {
        "input-tensor": [i for i in range(num_cubes_in_input_tensor)],
        "filter-tensor": [
            i + num_cubes_in_input_tensor
            for i in range(num_cubes_in_filter_tensor)
        ],
    }


def test_space_does_not_allow_duplicate_names() -> None:
    space = bb.Space()

    first_cube = bb.Cube(base_vector=np.array([0, 0, 0]), name="foo")
    second_cube = bb.Cube(base_vector=np.array([0, 0, 0]), name="foo")

    space.add_cube(first_cube)
    expected_err_msg = "There already exists an object with name foo."

    with pytest.raises(Exception, match=expected_err_msg):
        space.add_cube(second_cube)


def test_space_correctly_reorients_data() -> None:
    ...
