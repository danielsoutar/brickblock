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
    assert np.array_equal(space.cuboid_shapes, np.zeros((10, 3)))
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
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (np.ones((1, 3)), np.zeros((expected_num_entries - 1, 3))),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None],
        "linewidth": [0.1],
        "edgecolor": ["black"],
        "alpha": [0.0],
    }
    assert list(space.cuboid_index.primitives()) == [0]
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
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.ones((2, 3)),
                np.zeros((expected_num_entries - 2, 3)),
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
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [1]
    assert space.cuboid_index.get_primitives_by_scene(0) == [0]
    assert space.cuboid_index.get_primitives_by_scene(1) == [1]
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
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.ones((2, 3)),
                np.zeros((expected_num_entries - 2, 3)),
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
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [1]
    assert space.cuboid_index.get_primitives_by_scene(0) == [0, 1]
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
    assert np.array_equal(space.total, np.array([[2.0], [1.0], [1.5]]))
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
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(np.array([w, h, d]), (num_cubes, 3)),
                np.zeros((expected_num_entries - num_cubes, 3)),
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
    composite = slice(0, num_cubes)
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_scene(0) == [composite]
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
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.array([[4, 2, 6]]),
                np.zeros((expected_num_entries - 1, 3)),
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
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_scene(0) == [0]

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
        "input-tensor": [None, [slice(0, num_cubes_in_input_tensor)]],
        "filter-tensor": [
            None,
            [
                slice(
                    num_cubes_in_input_tensor,
                    num_cubes_in_input_tensor + num_cubes_in_filter_tensor,
                )
            ],
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


def test_space_mutates_primitive_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0

    space.mutate_by_coordinate(coordinate=point, facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={"facecolor": [None], "alpha": [0]}, coordinate=point
        ),
    ]

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_mutates_composite_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_composite(
        bb.CompositeCube(
            base_vector=point,
            w=4,
            h=3,
            d=2,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
        )
    )

    num_cubes = 4 * 3 * 2

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] == "yellow"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.3
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    space.mutate_by_coordinate(coordinate=point, facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": ["yellow"],
                "alpha": [0.3],
            },
            coordinate=point,
        ),
    ]

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] is None
        assert space.cuboid_visual_metadata["alpha"][i] == 0.0
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_mutates_multiple_objects_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_composite(
        bb.CompositeCube(
            base_vector=point,
            w=4,
            h=3,
            d=2,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
        )
    )

    num_cubes = 4 * 3 * 2

    space.add_cube(bb.Cube(base_vector=point))

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] == "yellow"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.3
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    assert space.cuboid_visual_metadata["facecolor"][-1] is None
    assert space.cuboid_visual_metadata["alpha"][-1] == 0.0
    assert space.cuboid_visual_metadata["linewidth"][-1] == 0.1

    space.mutate_by_coordinate(coordinate=point, facecolor="red", alpha=1.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Addition(1, None),
        bb.Mutation(
            subject={
                "facecolor": [None, "yellow"],
                "alpha": [0.0, 0.3],
            },
            coordinate=point,
        ),
    ]

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] == "red"
        assert space.cuboid_visual_metadata["alpha"][i] == 1.0
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    assert space.cuboid_visual_metadata["facecolor"][-1] == "red"
    assert space.cuboid_visual_metadata["alpha"][-1] == 1.0
    assert space.cuboid_visual_metadata["linewidth"][-1] == 0.1

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.primitives()) == [num_cubes, num_cubes]
    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [num_cubes]
    assert space.cuboid_index.get_primitives_by_timestep(2) == [num_cubes]
    assert space.cuboid_index.get_composites_by_timestep(2) == [composite]


def test_space_mutates_primitive_by_name() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0]), name="my-cube"))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0
    assert list(space.cuboid_names.keys()) == ["my-cube"]

    space.mutate_by_name(name="my-cube", facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": [None],
                "alpha": [0.0],
            },
            name="my-cube",
        ),
    ]

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert list(space.cuboid_names.keys()) == ["my-cube"]

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_mutates_composite_by_name() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
            name="my-composite",
        )
    )

    num_cubes = 4 * 3 * 2

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] == "yellow"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.3
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    space.mutate_by_name(name="my-composite", facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": ["yellow"],
                "alpha": [0.3],
            },
            name="my-composite",
        ),
    ]

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] is None
        assert space.cuboid_visual_metadata["alpha"][i] == 0.0
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_mutates_primitive_by_timestep_id() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0])))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0

    space.mutate_by_timestep(timestep=0, facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": [None],
                "alpha": [0.0],
            },
            timestep_id=0,
        ),
    ]

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_mutates_composite_by_timestep_id() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
        )
    )

    num_cubes = 4 * 3 * 2

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] == "yellow"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.3
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    space.mutate_by_timestep(timestep=0, facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": ["yellow"],
                "alpha": [0.3],
            },
            timestep_id=0,
        ),
    ]

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] is None
        assert space.cuboid_visual_metadata["alpha"][i] == 0.0
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_mutates_primitive_by_scene_id() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0])))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0

    space.mutate_by_scene(scene=0, facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": [None],
                "alpha": [0.0],
            },
            scene_id=0,
        ),
    ]

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_mutates_composite_by_scene_id() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
        )
    )

    num_cubes = 4 * 3 * 2

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] == "yellow"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.3
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    space.mutate_by_scene(scene=0, facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Mutation(
            subject={
                "facecolor": ["yellow"],
                "alpha": [0.3],
            },
            scene_id=0,
        ),
    ]

    for i in range(num_cubes):
        assert space.cuboid_visual_metadata["facecolor"][i] is None
        assert space.cuboid_visual_metadata["alpha"][i] == 0.0
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_mutates_multiple_objects_by_scene_id() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor=None,
            alpha=0.3,
            linewidth=0.5,
            name="input-tensor",
        )
    )

    space.add_cube(bb.Cube(base_vector=np.array([12, 14, 3])))

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=3,
            h=3,
            d=2,
            facecolor="red",
            alpha=0.5,
            linewidth=0.7,
            name="filter-tensor",
        )
    )

    # Check that only the first scene is affected.
    space.snapshot()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([3, 3, 3]),
            w=5,
            h=5,
            d=2,
            facecolor="orange",
            alpha=0.6,
            linewidth=0.8,
            name="unchanged-tensor",
        )
    )

    num_cubes_input_tensor = 4 * 3 * 2
    num_cubes_filter_tensor = 3 * 3 * 2
    num_cubes_unchanged_tensor = 5 * 5 * 2

    for i in range(num_cubes_input_tensor):
        assert space.cuboid_visual_metadata["facecolor"][i] is None
        assert space.cuboid_visual_metadata["alpha"][i] == 0.3
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.5

    offset = num_cubes_input_tensor

    assert space.cuboid_visual_metadata["facecolor"][offset] is None
    assert space.cuboid_visual_metadata["alpha"][offset] == 0
    assert space.cuboid_visual_metadata["linewidth"][offset] == 0.1

    for i in range(num_cubes_filter_tensor):
        offset = num_cubes_input_tensor + 1
        assert space.cuboid_visual_metadata["facecolor"][offset + i] == "red"
        assert space.cuboid_visual_metadata["alpha"][offset + i] == 0.5
        assert space.cuboid_visual_metadata["linewidth"][offset + i] == 0.7

    for i in range(num_cubes_unchanged_tensor):
        offset = num_cubes_input_tensor + 1 + num_cubes_filter_tensor
        assert space.cuboid_visual_metadata["facecolor"][offset + i] == "orange"
        assert space.cuboid_visual_metadata["alpha"][offset + i] == 0.6
        assert space.cuboid_visual_metadata["linewidth"][offset + i] == 0.8

    space.mutate_by_scene(scene=0, facecolor="black", alpha=0.9, linewidth=0.1)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Addition(1, None),
        bb.Addition(2, None),
        bb.Addition(3, None),
        bb.Mutation(
            subject={
                "facecolor": [None, None, "red"],
                "alpha": [0.0, 0.3, 0.5],
                "linewidth": [0.1, 0.5, 0.7],
            },
            scene_id=0,
        ),
    ]

    for i in range(num_cubes_input_tensor + num_cubes_filter_tensor + 1):
        assert space.cuboid_visual_metadata["facecolor"][i] == "black"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.9
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.1

    for i in range(num_cubes_unchanged_tensor):
        offset = num_cubes_input_tensor + num_cubes_filter_tensor + 1
        assert space.cuboid_visual_metadata["facecolor"][offset + i] == "orange"
        assert space.cuboid_visual_metadata["alpha"][offset + i] == 0.6
        assert space.cuboid_visual_metadata["linewidth"][offset + i] == 0.8

    input_N = num_cubes_input_tensor + 1
    filter_N = num_cubes_filter_tensor
    unchanged_N = num_cubes_unchanged_tensor

    first_composite = slice(0, input_N - 1)
    second_composite = slice(input_N, input_N + filter_N)
    last_composite = slice(input_N + filter_N, input_N + filter_N + unchanged_N)

    assert list(space.cuboid_index.primitives()) == [input_N - 1, input_N - 1]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [input_N - 1]
    assert space.cuboid_index.get_primitives_by_timestep(4) == [input_N - 1]
    assert space.cuboid_index.get_primitives_by_scene(0) == [input_N - 1]
    assert space.cuboid_index.get_primitives_by_scene(1) == [input_N - 1]

    assert list(space.cuboid_index.composites()) == [
        first_composite,
        second_composite,
        last_composite,
        first_composite,
        second_composite,
    ]
    assert space.cuboid_index.get_composites_by_timestep(4) == [
        first_composite,
        second_composite,
    ]
    assert space.cuboid_index.get_composites_by_scene(0) == [
        first_composite,
        second_composite,
    ]
    assert space.cuboid_index.get_composites_by_scene(1) == [
        last_composite,
        first_composite,
        second_composite,
    ]


def test_space_mutates_multiple_objects_multiple_times() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor=None,
            alpha=0.3,
            linewidth=0.5,
            name="input-tensor",
        )
    )

    space.add_cube(bb.Cube(base_vector=np.array([12, 14, 3])))

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=3,
            h=3,
            d=2,
            facecolor="red",
            alpha=0.5,
            linewidth=0.7,
            name="filter-tensor",
        )
    )

    # Check that only the first scene is affected.
    space.snapshot()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([3, 3, 3]),
            w=5,
            h=5,
            d=2,
            facecolor="orange",
            alpha=0.6,
            linewidth=0.8,
            name="unchanged-tensor",
        )
    )

    space.mutate_by_scene(scene=0, facecolor="black", alpha=0.9, linewidth=0.1)
    space.mutate_by_scene(scene=1, edgecolor="red")
    space.mutate_by_name(name="input-tensor", facecolor="white")

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Addition(1, None),
        bb.Addition(2, None),
        bb.Addition(3, None),
        bb.Mutation(
            subject={
                "facecolor": [None, None, "red"],
                "alpha": [0.0, 0.3, 0.5],
                "linewidth": [0.1, 0.5, 0.7],
            },
            scene_id=0,
        ),
        bb.Mutation(
            subject={"edgecolor": ["black", "black", "black", "black"]},
            scene_id=1,
        ),
        bb.Mutation(
            subject={"facecolor": ["black"]},
            name="input-tensor",
        ),
    ]


def test_space_mutates_nothing_on_empty_selection() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor=None,
            alpha=0.3,
            linewidth=0.5,
            name="input-tensor",
        )
    )

    space.add_cube(bb.Cube(base_vector=np.array([12, 14, 3])))

    # Check that only the first scene is affected.
    space.snapshot()

    expected_name_err_msg = "The provided name does not exist in this space."
    with pytest.raises(Exception, match=expected_name_err_msg):
        space.mutate_by_name("not-a-valid-name", facecolor="black")

    expected_timestep_err_msg = (
        "The provided timestep is invalid in this space."
    )
    with pytest.raises(Exception, match=expected_timestep_err_msg):
        space.mutate_by_timestep(timestep=3, edgecolor="white")

    # The scene is technically valid (it is the current scene), but it is empty,
    # so it silently does nothing.
    space.mutate_by_scene(scene=1, linewidth=0.69)

    # An error isn't meaningful in this case, so it silently does nothing.
    space.mutate_by_coordinate(np.array([64, 32, 16]), alpha=1.0, linewidth=1.0)

    # None of the above mutations have any effect and are not reflected in the
    # history.
    assert space.changelog == [
        bb.Addition(timestep_id=0, name=None),
        bb.Addition(timestep_id=1, name=None),
    ]


def test_space_updates_bounds_with_cube() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([-1, -2, -3]), scale=2.0))

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    expected_dims = np.array([[-1, 1], [-3, -1], [-2, 0]])

    assert np.array_equal(space.dims, expected_dims)


def test_space_updates_bounds_with_cuboid() -> None:
    space = bb.Space()

    space.add_cube(
        bb.Cuboid(base_vector=np.array([-3, -2, -1]), w=4.0, h=15.0, d=26.0)
    )

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    expected_dims = np.array([[-3, 1], [-1, 25], [-2, 13]])

    assert np.array_equal(space.dims, expected_dims)


def test_space_updates_bounds_with_composite() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(base_vector=np.array([1, 5, 10]), w=8, h=12, d=3)
    )

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    expected_dims = np.array([[1, 9], [10, 13], [5, 17]])

    assert np.array_equal(space.dims, expected_dims)


# TODO: This is very slow due to the composite (~14,000 cubes) - make fast.
def test_space_updates_bounds_with_multiple_objects() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0]), scale=2.0))

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    expected_dims = np.array([[0, 2], [0, 2], [0, 2]])

    assert np.array_equal(space.dims, expected_dims)

    space.add_cuboid(
        bb.Cuboid(base_vector=np.array([100, 100, 100]), w=5, h=6, d=13)
    )

    expected_dims = np.array([[0, 105], [0, 113], [0, 106]])

    assert np.array_equal(space.dims, expected_dims)

    space.add_composite(
        bb.CompositeCube(base_vector=np.array([30, 30, 30]), w=40, h=30, d=12)
    )

    # The extrema of the space should not have changed.
    assert np.array_equal(space.dims, expected_dims)


def test_space_creates_cuboid_from_offset_with_selections() -> None:
    space = bb.Space()

    base_point = np.array([0, 0, 0])
    space.add_cube(bb.Cube(base_vector=base_point, scale=2.0, name="my-cube"))

    space.clone_by_offset(np.array([12, 0, 0]), coordinate=base_point)
    space.clone_by_offset(np.array([0, 12, 0]), name="my-cube")
    space.clone_by_offset(np.array([0, 0, 12]), timestep=0)
    space.snapshot()
    space.clone_by_offset(np.array([32, 0, 0]), scene=0)

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    expected_dims = np.array([[0, 46], [0, 14], [0, 14]])
    assert np.array_equal(space.dims, expected_dims)
    expected_mean = np.array([[20], [4], [4]])
    assert np.array_equal(space.mean, expected_mean)
    # TODO: Remove the total field as it's not yet needed (+ probably won't be).
    expected_total = np.array([[160], [32], [32]])
    assert np.array_equal(space.total, expected_total)
    assert space.num_objs == 8
    assert space.primitive_counter == 8
    assert space.time_step == 5
    assert space.scene_counter == 1
    expected_num_entries = 8
    expected_coordinate_entry = mock_coordinates_entry() * 2
    sup_width_offset = np.array([32, 0, 0])
    sub_width_offset = np.array([12, 0, 0])
    height_offset = np.array([0, 0, 12])
    depth_offset = np.array([0, 12, 0])

    expected_coordinates = np.concatenate(
        (
            expected_coordinate_entry,
            expected_coordinate_entry + sub_width_offset,
            expected_coordinate_entry + height_offset,
            expected_coordinate_entry + depth_offset,
            expected_coordinate_entry + sup_width_offset,
            expected_coordinate_entry + sup_width_offset + sub_width_offset,
            expected_coordinate_entry + sup_width_offset + height_offset,
            expected_coordinate_entry + sup_width_offset + depth_offset,
            np.zeros((10 - expected_num_entries, 6, 4, 3)),
        ),
        axis=0,
    )
    assert np.array_equal(space.cuboid_coordinates, expected_coordinates)
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.full((expected_num_entries, 3), 2),
                np.zeros((10 - expected_num_entries, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None] * expected_num_entries,
        "linewidth": [0.1] * expected_num_entries,
        "edgecolor": ["black"] * expected_num_entries,
        "alpha": [0.0] * expected_num_entries,
    }
    assert list(space.cuboid_index.primitives()) == [
        i for i in range(expected_num_entries)
    ]
    assert space.changelog == [bb.Addition(i, None) for i in range(5)]
    assert space.cuboid_index.get_primitives_by_timestep(4) == [4, 5, 6, 7]


def test_space_creates_composites_from_offset_with_selections() -> None:
    space = bb.Space()

    base_point = np.array([0, 0, 0])
    w, h, d = 3, 4, 2
    space.add_composite(
        bb.CompositeCube(
            base_vector=base_point, w=w, h=h, d=d, name="my-composite"
        )
    )

    space.clone_by_offset(np.array([12, 0, 0]), coordinate=base_point)
    # This should be treated as a no-op.
    space.clone_by_offset(np.array([12, 0, 0]), coordinate=base_point + 37)
    space.clone_by_offset(np.array([0, 12, 0]), name="my-composite")
    space.clone_by_offset(np.array([0, 0, 12]), timestep=0)
    space.snapshot()
    space.clone_by_offset(np.array([32, 0, 0]), scene=0)

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    expected_dims = np.array([[0, 47], [0, 14], [0, 16]])
    assert np.array_equal(space.dims, expected_dims)
    expected_mean = np.array([[20.5], [4], [5]])
    assert np.array_equal(space.mean, expected_mean)
    # TODO: Remove the total field as it's not yet needed (+ probably won't be).
    expected_total = np.array([[164], [32], [40]])
    assert np.array_equal(space.total, expected_total)
    assert space.num_objs == 8
    num_cubes_per_composite = w * h * d
    assert space.primitive_counter == num_cubes_per_composite * 8
    assert space.time_step == 5
    assert space.scene_counter == 1
    expected_num_entries = num_cubes_per_composite * 8
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    expected_coordinate_entry = np.squeeze(
        np.array(
            [
                mock_coordinates_entry()
                + (w * width)
                + (h * height)
                + (d * depth)
                for (w, h, d) in itertools.product(range(w), range(h), range(d))
            ]
        )
    )
    sup_width_offset = np.array([32, 0, 0])
    sub_width_offset = np.array([12, 0, 0])
    height_offset = np.array([0, 0, 12])
    depth_offset = np.array([0, 12, 0])

    expected_coordinates = np.concatenate(
        (
            expected_coordinate_entry,
            expected_coordinate_entry + sub_width_offset,
            expected_coordinate_entry + height_offset,
            expected_coordinate_entry + depth_offset,
            expected_coordinate_entry + sup_width_offset,
            expected_coordinate_entry + sup_width_offset + sub_width_offset,
            expected_coordinate_entry + sup_width_offset + height_offset,
            expected_coordinate_entry + sup_width_offset + depth_offset,
            np.zeros((320 - expected_num_entries, 6, 4, 3)),
        ),
        axis=0,
    )
    assert np.array_equal(space.cuboid_coordinates, expected_coordinates)
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(
                    np.array([[3, 4, 2]]), (expected_num_entries, 3)
                ),
                np.zeros((320 - expected_num_entries, 3)),
            ),
            axis=0,
        ),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None] * expected_num_entries,
        "linewidth": [0.1] * expected_num_entries,
        "edgecolor": ["black"] * expected_num_entries,
        "alpha": [0.0] * expected_num_entries,
    }
    assert list(space.cuboid_index.composites()) == [
        slice(i, i + 24) for i in range(0, 192, 24)
    ]
    assert space.changelog == [bb.Addition(i, None) for i in range(5)]
    assert space.cuboid_index.get_composites_by_timestep(4) == [
        slice(4 * 24, 5 * 24),
        slice(5 * 24, 6 * 24),
        slice(6 * 24, 7 * 24),
        slice(7 * 24, 8 * 24),
    ]


def test_space_create_from_offset_only_uses_one_selection() -> None:
    space = bb.Space()

    base_point = np.array([0, 0, 0])
    w, h, d = 3, 4, 2
    space.add_composite(
        bb.CompositeCube(
            base_vector=base_point, w=w, h=h, d=d, name="my-composite"
        )
    )

    expected_err_msg = (
        "Exactly one selection argument can be set when creating objects."
    )
    with pytest.raises(Exception, match=expected_err_msg):
        space.clone_by_offset(
            np.array([12, 0, 0]), coordinate=base_point, name="my-composite"
        )

    # TODO: Consider whether to support this case - and whether named objects
    # should support mutation of an earlier iteration.
    with pytest.raises(Exception, match=expected_err_msg):
        space.clone_by_offset(
            np.array([12, 0, 0]), timestep=0, name="my-composite"
        )


def test_space_creates_composites_from_offset_with_updated_visuals() -> None:
    space = bb.Space()

    base_point = np.array([0, 0, 0])
    w, h, d = 3, 4, 2
    space.add_composite(
        bb.CompositeCube(
            base_vector=base_point, w=w, h=h, d=d, name="my-composite"
        )
    )

    # Check the scalar case.
    space.clone_by_offset(
        np.array([12, 0, 0]),
        coordinate=base_point,
        facecolor="blue",
        edgecolor="white",
        alpha=0.2,
    )
    # This should be treated as a no-op.
    space.clone_by_offset(np.array([12, 0, 0]), coordinate=base_point + 37)
    space.clone_by_offset(np.array([0, 12, 0]), name="my-composite")
    space.clone_by_offset(np.array([0, 0, 12]), timestep=0)
    space.snapshot()
    # Check the iterable case.
    num_cubes_per_composite = w * h * d
    new_face_colors = ["black", "brown", "green", "blue"]
    new_edge_colors = ["white", "black", "purple", "white"]
    all_face_colors = [
        primitive_fc
        for fc in [None, "blue", None, None] + new_face_colors
        for primitive_fc in [fc] * num_cubes_per_composite
    ]
    all_edge_colors = [
        primitive_ec
        for ec in ["black", "white", "black", "black"] + new_edge_colors
        for primitive_ec in [ec] * num_cubes_per_composite
    ]
    # By default the alpha should be 1.0 when face colors are set, but we make
    # that explicit here for clarity.
    space.clone_by_offset(
        np.array([32, 0, 0]),
        scene=0,
        facecolor=new_face_colors,
        edgecolor=new_edge_colors,
        alpha=[1.0] * 4,
    )

    assert space.cuboid_visual_metadata == {
        "facecolor": all_face_colors,
        "linewidth": [0.1] * (8 * num_cubes_per_composite),
        "edgecolor": all_edge_colors,
        "alpha": ([0.0] * num_cubes_per_composite)
        + ([0.2] * num_cubes_per_composite)
        + [0.0] * (2 * num_cubes_per_composite)
        + [1.0] * (4 * num_cubes_per_composite),
    }


def test_space_transforms_primitive_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    space.transform_by_coordinate(coordinate=point, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate, transform_name="translation", coordinate=point
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])

    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry() + swapped_point + translate,
                np.zeros((expected_num_entries - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.array([[1, 1, 1]]),
                np.zeros((expected_num_entries - 1, 3)),
            ),
            axis=0,
        ),
    )

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_transforms_composite_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(
        bb.CompositeCube(
            base_vector=point,
            w=w,
            h=h,
            d=d,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
        )
    )

    num_cubes = w * h * d

    translate = np.array([3, 3, 3])
    space.transform_by_coordinate(coordinate=point, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            coordinate=point,
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    def make_sub_cubes():
        return [
            mock_coordinates_entry()
            + swapped_point
            + translate
            + (w * width)
            + (h * height)
            + (d * depth)
            for (w, h, d) in itertools.product(range(w), range(h), range(d))
        ]

    expected_num_entries = 40
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                *make_sub_cubes(),
                np.zeros((expected_num_entries - num_cubes, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(np.array([[w, h, d]]), (num_cubes, 3)),
                np.zeros((expected_num_entries - num_cubes, 3)),
            ),
            axis=0,
        ),
    )

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_transforms_multiple_objects_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(
        bb.CompositeCube(
            base_vector=point,
            w=w,
            h=h,
            d=d,
            facecolor="yellow",
            alpha=0.3,
            linewidth=0.5,
        )
    )

    num_cubes = w * h * d

    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    space.transform_by_coordinate(coordinate=point, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Addition(1, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            coordinate=point,
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    def make_sub_cubes():
        return [
            mock_coordinates_entry()
            + swapped_point
            + translate
            + (w * width)
            + (h * height)
            + (d * depth)
            for (w, h, d) in itertools.product(range(w), range(h), range(d))
        ]

    expected_num_entries = 40
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                *make_sub_cubes(),
                mock_coordinates_entry() + swapped_point + translate,
                np.zeros((expected_num_entries - num_cubes - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(np.array([[w, h, d]]), (num_cubes, 3)),
                np.array([[1, 1, 1]]),
                np.zeros((expected_num_entries - num_cubes - 1, 3)),
            ),
            axis=0,
        ),
    )

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.primitives()) == [num_cubes, num_cubes]
    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [num_cubes]
    assert space.cuboid_index.get_primitives_by_timestep(2) == [num_cubes]
    assert space.cuboid_index.get_composites_by_timestep(2) == [composite]


def test_space_transforms_primitive_by_name() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point, name="my-primitive"))

    translate = np.array([3, 3, 3])
    space.transform_by_name(name="my-primitive", translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            name="my-primitive",
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])

    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry() + swapped_point + translate,
                np.zeros((expected_num_entries - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.array([[1, 1, 1]]),
                np.zeros((expected_num_entries - 1, 3)),
            ),
            axis=0,
        ),
    )

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_transforms_composite_by_name() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(
        bb.CompositeCube(base_vector=point, w=w, h=h, d=d, name="my-composite")
    )

    num_cubes = w * h * d

    translate = np.array([3, 3, 3])
    space.transform_by_name(name="my-composite", translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            name="my-composite",
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    def make_sub_cubes():
        return [
            mock_coordinates_entry()
            + swapped_point
            + translate
            + (w * width)
            + (h * height)
            + (d * depth)
            for (w, h, d) in itertools.product(range(w), range(h), range(d))
        ]

    expected_num_entries = 40
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                *make_sub_cubes(),
                np.zeros((expected_num_entries - num_cubes, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(np.array([[w, h, d]]), (num_cubes, 3)),
                np.zeros((expected_num_entries - num_cubes, 3)),
            ),
            axis=0,
        ),
    )

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_transforms_primitive_by_timestep_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    space.transform_by_timestep(timestep=0, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            timestep_id=0,
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])

    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry() + swapped_point + translate,
                np.zeros((expected_num_entries - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.array([[1, 1, 1]]),
                np.zeros((expected_num_entries - 1, 3)),
            ),
            axis=0,
        ),
    )

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_transforms_composite_by_timestep_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(bb.CompositeCube(base_vector=point, w=w, h=h, d=d))

    num_cubes = w * h * d

    translate = np.array([3, 3, 3])
    space.transform_by_timestep(timestep=0, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            timestep_id=0,
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    def make_sub_cubes():
        return [
            mock_coordinates_entry()
            + swapped_point
            + translate
            + (w * width)
            + (h * height)
            + (d * depth)
            for (w, h, d) in itertools.product(range(w), range(h), range(d))
        ]

    expected_num_entries = 40
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                *make_sub_cubes(),
                np.zeros((expected_num_entries - num_cubes, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(np.array([[w, h, d]]), (num_cubes, 3)),
                np.zeros((expected_num_entries - num_cubes, 3)),
            ),
            axis=0,
        ),
    )

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_transforms_primitive_by_scene_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    space.transform_by_scene(scene=0, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            scene_id=0,
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])

    expected_num_entries = 10
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                mock_coordinates_entry() + swapped_point + translate,
                np.zeros((expected_num_entries - 1, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.array([[1, 1, 1]]),
                np.zeros((expected_num_entries - 1, 3)),
            ),
            axis=0,
        ),
    )

    assert list(space.cuboid_index.primitives()) == [0, 0]
    assert space.cuboid_index.get_primitives_by_timestep(0) == [0]
    assert space.cuboid_index.get_primitives_by_timestep(1) == [0]


def test_space_transforms_composite_by_scene_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(bb.CompositeCube(base_vector=point, w=w, h=h, d=d))

    num_cubes = w * h * d

    translate = np.array([3, 3, 3])
    space.transform_by_scene(scene=0, translate=translate)

    # Check the changelog reflects the transform, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            scene_id=0,
        ),
    ]

    # Remember to swap the ys and zs due to the current implementation issue
    # with dims
    # TODO: Have a transform for matplotlib and have your own representation
    # instead.
    swapped_point = np.array([1, 3, 2])
    width = np.array([1, 0, 0])
    height = np.array([0, 0, 1])
    depth = np.array([0, 1, 0])

    def make_sub_cubes():
        return [
            mock_coordinates_entry()
            + swapped_point
            + translate
            + (w * width)
            + (h * height)
            + (d * depth)
            for (w, h, d) in itertools.product(range(w), range(h), range(d))
        ]

    expected_num_entries = 40
    assert np.array_equal(
        space.cuboid_coordinates,
        np.concatenate(
            (
                *make_sub_cubes(),
                np.zeros((expected_num_entries - num_cubes, 6, 4, 3)),
            ),
            axis=0,
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(np.array([[w, h, d]]), (num_cubes, 3)),
                np.zeros((expected_num_entries - num_cubes, 3)),
            ),
            axis=0,
        ),
    )

    composite = slice(0, num_cubes)

    assert list(space.cuboid_index.composites()) == [composite, composite]
    assert space.cuboid_index.get_composites_by_timestep(0) == [composite]
    assert space.cuboid_index.get_composites_by_timestep(1) == [composite]


def test_space_transforms_multiple_objects_multiple_times() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor=None,
            alpha=0.3,
            linewidth=0.5,
            name="input-tensor",
        )
    )

    space.add_cube(bb.Cube(base_vector=np.array([12, 14, 3])))

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=3,
            h=3,
            d=2,
            facecolor="red",
            alpha=0.5,
            linewidth=0.7,
            name="filter-tensor",
        )
    )

    # Check that only the first scene is affected.
    space.snapshot()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([3, 3, 3]),
            w=5,
            h=5,
            d=2,
            facecolor="orange",
            alpha=0.6,
            linewidth=0.8,
            name="unchanged-tensor",
        )
    )

    first_translate = np.array([4, 5, 6])
    second_translate = np.array([10, 14, 2])
    third_translate = np.array([300, 200, 100])
    space.transform_by_scene(scene=0, translate=first_translate)
    space.transform_by_scene(scene=1, translate=second_translate)
    space.transform_by_name(name="input-tensor", translate=third_translate)

    # Check the changelog reflects the transforms, storing the previous state.
    assert space.changelog == [
        bb.Addition(0, None),
        bb.Addition(1, None),
        bb.Addition(2, None),
        bb.Addition(3, None),
        bb.Transform(
            transform=-first_translate,
            transform_name="translation",
            scene_id=0,
        ),
        bb.Transform(
            transform=-second_translate,
            transform_name="translation",
            scene_id=1,
        ),
        bb.Transform(
            transform=-third_translate,
            transform_name="translation",
            name="input-tensor",
        ),
    ]


def test_space_transforms_nothing_on_empty_selection() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=4,
            h=3,
            d=2,
            facecolor=None,
            alpha=0.3,
            linewidth=0.5,
            name="input-tensor",
        )
    )

    space.add_cube(bb.Cube(base_vector=np.array([12, 14, 3])))

    # Check that only the first scene is affected.
    space.snapshot()

    translate = np.array([3, 3, 3])

    expected_name_err_msg = "The provided name does not exist in this space."
    with pytest.raises(Exception, match=expected_name_err_msg):
        space.transform_by_name("not-a-valid-name", translate=translate)

    expected_timestep_err_msg = (
        "The provided timestep is invalid in this space."
    )
    with pytest.raises(Exception, match=expected_timestep_err_msg):
        space.transform_by_timestep(timestep=3, translate=translate)

    # The scene is technically valid (it is the current scene), but it is empty,
    # so it silently does nothing.
    space.transform_by_scene(scene=1, translate=translate)

    # An error isn't meaningful in this case, so it silently does nothing.
    space.transform_by_coordinate(np.array([64, 32, 16]), translate=translate)

    # None of the above transforms have any effect and are not reflected in the
    # history.
    assert space.changelog == [
        bb.Addition(timestep_id=0, name=None),
        bb.Addition(timestep_id=1, name=None),
    ]


def test_space_supports_composites_with_classic_style() -> None:
    ...


def test_space_correctly_reorients_data() -> None:
    ...
