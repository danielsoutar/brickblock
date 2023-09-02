import pytest

import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np

import brickblock as bb
import brickblock.visualisation as bb_vis


def test_space_creation() -> None:
    space = bb.Space()

    assert np.array_equal(space.mean, np.zeros((3, 1)))
    assert np.array_equal(space.total, np.zeros((3, 1)))
    assert space.num_objs == 0
    assert space.object_counter == 0
    assert space.time_step == 0
    assert space.scene_counter == 0
    assert np.array_equal(space.base_coordinates, np.zeros((10, 3)))
    assert np.array_equal(space.cuboid_shapes, np.zeros((10, 3)))
    assert space.cuboid_visual_metadata == {}
    assert space.cuboid_index is not None
    assert space.composite_index is not None
    assert space.changelog == []


def test_space_snapshot_creates_a_scene() -> None:
    space = bb.Space()

    point = np.array([0, 0, 0]).reshape((1, 3))
    cube = bb.Cube(base_vector=point, scale=1.0)
    space.add_cube(cube)
    space.snapshot()

    assert np.array_equal(space.mean, np.array([[0.5], [0.5], [0.5]]))
    assert np.array_equal(space.total, np.array([[0.5], [0.5], [0.5]]))
    assert space.num_objs == 1
    assert space.object_counter == 1
    assert space.time_step == 1
    assert space.scene_counter == 1
    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((np.ones((1, 3)), np.zeros((empty_entries, 3))), axis=0),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None],
        "linewidth": [0.1],
        "edgecolor": ["black"],
        "alpha": [0.0],
    }
    assert list(space.cuboid_index.items()) == [0]
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        )
    ]


def test_space_multiple_snapshots_create_multiple_scenes() -> None:
    space = bb.Space()

    point = np.array([0, 0, 0]).reshape((1, 3))
    cube = bb.Cube(base_vector=point, scale=1.0)
    space.add_cube(cube)
    space.snapshot()

    second_cube = bb.Cube(base_vector=point + 3, scale=1.0)
    space.add_cube(second_cube)
    space.snapshot()

    assert np.array_equal(space.mean, np.array([[2.0], [2.0], [2.0]]))
    assert np.array_equal(space.total, np.array([[4.0], [4.0], [4.0]]))
    assert space.num_objs == 2
    assert space.object_counter == 2
    assert space.time_step == 2
    assert space.scene_counter == 2
    empty_entries = 8
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate(
            (point, point + 3, np.zeros((empty_entries, 3))), axis=0
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((np.ones((2, 3)), np.zeros((empty_entries, 3))), axis=0),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None, None],
        "linewidth": [0.1, 0.1],
        "edgecolor": ["black", "black"],
        "alpha": [0.0, 0.0],
    }
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [1]
    assert space.cuboid_index.get_items_by_scene(0) == [0]
    assert space.cuboid_index.get_items_by_scene(1) == [1]
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
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

    point = np.array([0, 0, 0])
    cube = bb.Cube(base_vector=point, scale=1.0)
    space.add_cube(cube)
    space.snapshot()
    _, ax = space.render()

    # Check view limits are correctly calculated with the extrema of the space.
    assert (ax.axes.xy_viewLim.x0, ax.axes.xy_viewLim.x1) == (-1, 1)
    assert (ax.axes.xy_viewLim.y0, ax.axes.xy_viewLim.y1) == (-1, 1)
    assert (ax.axes.zz_viewLim.x0, ax.axes.zz_viewLim.x1) == (-1, 1)

    plt_internal_data = ax.collections[0]._vec
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    cube_shape = np.array(cube.shape())
    cube_faces = bb_vis.materialise_vertices_for_primitive(point, cube_shape)

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cube_faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_space_does_nothing_on_render_when_empty() -> None:
    ...


def test_space_creates_valid_axes_on_render_multiple_scenes() -> None:
    space = bb.Space()

    point = np.array([0, 0, 0])
    cube = bb.Cube(base_vector=point, scale=1.0)
    space.add_cube(cube)
    space.snapshot()
    # Check this runs without issues, but we don't need the fig for this test.
    space.render()

    second_cube = bb.Cube(base_vector=point + 3, scale=1.0)
    space.add_cube(second_cube)
    space.snapshot()
    _, ax2 = space.render()

    # Check view limits are correctly calculated with the extrema of the space.
    assert (ax2.axes.xy_viewLim.x0, ax2.axes.xy_viewLim.x1) == (-4, 4)
    assert (ax2.axes.xy_viewLim.y0, ax2.axes.xy_viewLim.y1) == (-4, 4)
    assert (ax2.axes.zz_viewLim.x0, ax2.axes.zz_viewLim.x1) == (-4, 4)

    plt_internal_data_for_first_cube = ax2.collections[0]._vec.T
    plt_internal_data_for_second_cube = ax2.collections[1]._vec.T
    plt_internal_reshaped_data = np.concatenate(
        [plt_internal_data_for_first_cube, plt_internal_data_for_second_cube],
        axis=0,
    ).reshape((2, 6, 4, 4))

    cube_shape = np.array(cube.shape())
    cube_faces = bb_vis.materialise_vertices_for_primitive(point, cube_shape)
    second_cube_faces = bb_vis.materialise_vertices_for_primitive(
        point + 3, cube_shape
    )

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_first_cube = np.concatenate([cube_faces, ones], -1)
    original_augmented_second_cube = np.concatenate(
        [second_cube_faces, ones], -1
    )

    expected_data = np.stack(
        [original_augmented_first_cube, original_augmented_second_cube], axis=0
    )

    assert np.array_equal(expected_data, plt_internal_reshaped_data)


def test_space_add_multiple_cubes_in_single_scene() -> None:
    space = bb.Space()

    point = np.array([0, 0, 0]).reshape((1, 3))
    cube = bb.Cube(base_vector=point, scale=1.0)
    second_cube = bb.Cube(base_vector=point + 3, scale=1.0)

    space.add_cube(cube)
    space.add_cube(second_cube)
    space.snapshot()

    assert space.num_objs == 2
    assert space.object_counter == 2
    assert space.time_step == 2
    assert space.scene_counter == 1

    empty_entries = 8
    empty_entries_arr = np.zeros((empty_entries, 3))

    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((point, point + 3, empty_entries_arr), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((np.ones((2, 3)), empty_entries_arr), axis=0),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None, None],
        "linewidth": [0.1, 0.1],
        "edgecolor": ["black", "black"],
        "alpha": [0.0, 0.0],
    }
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [1]
    assert space.cuboid_index.get_items_by_scene(0) == [0, 1]
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
    ]


def test_space_creates_valid_axes_on_render_multiple_cubes_single_scene() -> (
    None
):
    space = bb.Space()

    first_point = np.array([0, 0, 0])
    second_point = np.array([3, 2, 1])
    cube = bb.Cube(base_vector=first_point, scale=1.0)
    second_cube = bb.Cube(base_vector=second_point, scale=1.0)

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

    cube_shape = np.array(cube.shape())
    cube_faces = bb_vis.materialise_vertices_for_primitive(
        first_point, cube_shape
    )
    # Swap the non-symmetric ys and zs for matplotlib compatibility.
    second_point_swapped = np.array([3, 1, 2])
    second_cube_faces = bb_vis.materialise_vertices_for_primitive(
        second_point_swapped, cube_shape
    )

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_first_cube = np.concatenate([cube_faces, ones], -1)
    original_augmented_second_cube = np.concatenate(
        [second_cube_faces, ones], -1
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

    cube_shape = np.array(cube.shape())
    # Swap the ys and zs for matplotlib compatibility.
    cube_points = list(
        map(
            lambda arr: np.array([arr[0], arr[2], arr[1]]),
            [cube.base, second_cube.base, third_cube.base, fourth_cube.base],
        )
    )
    all_cube_faces = [
        bb_vis.materialise_vertices_for_primitive(cube_points[i], cube_shape)
        for i in range(4)
    ]

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_cubes = [
        np.concatenate([c, ones], -1) for c in all_cube_faces
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

    point = np.array([0, 0, 0]).reshape((1, 3))
    w, h, d = 4, 3, 2
    composite = bb.CompositeCube(base_vector=point, w=w, h=h, d=d)

    space.add_composite(composite)
    space.snapshot()

    assert np.array_equal(space.mean, np.array([[2.0], [1.5], [1.0]]))
    assert np.array_equal(space.total, np.array([[2.0], [1.5], [1.0]]))
    assert space.num_objs == 1
    assert space.object_counter == 1
    assert space.time_step == 1
    assert space.scene_counter == 1

    empty_entries = 9
    empty_entries_arr = np.zeros((empty_entries, 3))

    expected_shape = np.array([w, h, d]).reshape((1, 3))

    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((point, empty_entries_arr), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, empty_entries_arr), axis=0),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None],
        "linewidth": [0.1],
        "edgecolor": ["black"],
        "alpha": [0.0],
    }
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_scene(0) == [0]
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["composite"],
            creation_type="manual",
            object_names=None,
        ),
    ]


def test_space_creates_valid_axes_on_render_for_composite() -> None:
    space = bb.Space()

    w, h, d = 4, 3, 2
    num_cubes = w * h * d

    first_point = np.array([0, 0, 0])
    second_point = np.array([w, h, d])
    composite = bb.CompositeCube(
        base_vector=first_point,
        w=w,
        h=h,
        d=d,
        facecolor="red",
    )
    second_composite = bb.CompositeCube(
        base_vector=second_point,
        w=w,
        h=h,
        d=d,
        facecolor="green",
    )
    space.add_composite(composite)
    space.add_composite(second_composite)
    _, ax = space.render()

    composite_shape = np.array(composite.shape())
    composite_faces = bb_vis.materialise_vertices_for_composite(
        first_point, composite_shape
    )
    # Swap the non-symmetric ys and zs for matplotlib compatibility.
    second_point_swapped = np.array([w, d, h])
    second_composite_faces = bb_vis.materialise_vertices_for_composite(
        second_point_swapped, composite_shape
    )

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    for i in range(num_cubes):
        plt_internal_data = ax.collections[i]._vec
        plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

        original_augmented_data = np.concatenate([composite_faces[i], ones], -1)

        assert np.array_equal(
            original_augmented_data, plt_internal_reshaped_data
        )

    for i in range(num_cubes, 2 * num_cubes):
        plt_internal_data = ax.collections[i]._vec
        plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

        offset = i - num_cubes
        original_augmented_data = np.concatenate(
            [second_composite_faces[offset], ones], -1
        )

        assert np.array_equal(
            original_augmented_data, plt_internal_reshaped_data
        )


def test_space_can_add_cuboid() -> None:
    space = bb.Space()

    point = np.array([0, 0, 0]).reshape((1, 3))
    w, h, d = 4, 2, 6
    cuboid = bb.Cuboid(base_vector=point, w=w, h=h, d=d)

    space.add_cuboid(cuboid)
    space.snapshot()

    assert np.array_equal(space.mean, np.array([[2], [1], [3]]))
    assert np.array_equal(space.total, np.array([[2], [1], [3]]))
    assert space.num_objs == 1
    assert space.object_counter == 1
    assert space.time_step == 1
    assert space.scene_counter == 1

    empty_entries = 9
    expected_shape = np.array([[4, 2, 6]]).reshape((1, 3))

    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )
    assert space.cuboid_visual_metadata == {
        "facecolor": [None],
        "linewidth": [0.1],
        "edgecolor": ["black"],
        "alpha": [0.0],
    }
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_scene(0) == [0]

    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
    ]


def test_space_creates_valid_axes_on_render_for_cuboid() -> None:
    space = bb.Space()

    point = np.array([0, 0, 0])
    cuboid = bb.Cuboid(base_vector=point, w=4.0, h=2.0, d=6.0)
    space.add_cuboid(cuboid)
    space.snapshot()
    _, ax = space.render()

    plt_internal_data = ax.collections[0]._vec
    plt_internal_reshaped_data = plt_internal_data.T.reshape((6, 4, 4))

    cuboid_shape = np.array(cuboid.shape())
    cuboid_faces = bb_vis.materialise_vertices_for_primitive(
        point, cuboid_shape
    )

    # Add the implicit 4th dimension to the original data - all ones.
    ones = np.ones((6, 4, 1))
    original_augmented_data = np.concatenate([cuboid_faces, ones], -1)

    assert np.array_equal(original_augmented_data, plt_internal_reshaped_data)


def test_space_can_add_named_objects() -> None:
    space = bb.Space()

    w, h, d = 4, 4, 3
    f_w, f_h, f_d = 2, 2, 3

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
    space.render()

    assert space.cuboid_names == {
        "input-tensor": [None, [0]],
        "filter-tensor": [None, [1]],
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
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
        bb.Mutation(
            subject={"facecolor": [None], "alpha": [0]}, coordinate=point
        ),
    ]

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3

    assert list(space.cuboid_index.items()) == [0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]


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

    assert space.cuboid_visual_metadata["facecolor"][0] == "yellow"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    space.mutate_by_coordinate(coordinate=point, facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["composite"],
            creation_type="manual",
            object_names=None,
        ),
        bb.Mutation(
            subject={
                "facecolor": ["yellow"],
                "alpha": [0.3],
            },
            coordinate=point,
        ),
    ]

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    assert list(space.composite_index.items()) == [0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]


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

    space.add_cube(bb.Cube(base_vector=point))

    assert space.cuboid_visual_metadata["facecolor"][0] == "yellow"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    assert space.cuboid_visual_metadata["facecolor"][1] is None
    assert space.cuboid_visual_metadata["alpha"][1] == 0.0
    assert space.cuboid_visual_metadata["linewidth"][1] == 0.1

    space.mutate_by_coordinate(coordinate=point, facecolor="red", alpha=1.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": [None, "yellow"],
            "alpha": [0.0, 0.3],
        },
        coordinate=point,
    )

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 1.0
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    assert space.cuboid_visual_metadata["facecolor"][1] == "red"
    assert space.cuboid_visual_metadata["alpha"][1] == 1.0
    assert space.cuboid_visual_metadata["linewidth"][1] == 0.1

    assert list(space.cuboid_index.items()) == [1, 1]
    assert list(space.composite_index.items()) == [0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [1]
    assert space.cuboid_index.get_items_by_timestep(2) == [1]
    assert space.composite_index.get_items_by_timestep(2) == [0]


def test_space_mutates_primitive_by_name() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0]), name="my-cube"))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0
    assert list(space.cuboid_names.keys()) == ["my-cube"]

    space.mutate_by_name(name="my-cube", facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": [None],
            "alpha": [0.0],
        },
        name="my-cube",
    )

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert list(space.cuboid_names.keys()) == ["my-cube"]

    assert list(space.cuboid_index.items()) == [0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]


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

    assert space.cuboid_visual_metadata["facecolor"][0] == "yellow"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    space.mutate_by_name(name="my-composite", facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": ["yellow"],
            "alpha": [0.3],
        },
        name="my-composite",
    )

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    assert list(space.composite_index.items()) == [0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]


def test_space_mutates_primitive_by_timestep_id() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0])))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0

    space.mutate_by_timestep(timestep=0, facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": [None],
            "alpha": [0.0],
        },
        timestep_id=0,
    )

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3

    assert list(space.cuboid_index.items()) == [0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]


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

    assert space.cuboid_visual_metadata["facecolor"][0] == "yellow"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    space.mutate_by_timestep(timestep=0, facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": ["yellow"],
            "alpha": [0.3],
        },
        timestep_id=0,
    )

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    assert list(space.composite_index.items()) == [0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]


def test_space_mutates_primitive_by_scene_id() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([0, 0, 0])))

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0

    space.mutate_by_scene(scene=0, facecolor="red", alpha=0.3)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": [None],
            "alpha": [0.0],
        },
        scene_id=0,
    )

    assert space.cuboid_visual_metadata["facecolor"][0] == "red"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3

    assert list(space.cuboid_index.items()) == [0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]


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

    assert space.cuboid_visual_metadata["facecolor"][0] == "yellow"
    assert space.cuboid_visual_metadata["alpha"][0] == 0.3
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    space.mutate_by_scene(scene=0, facecolor=None, alpha=0.0)

    # Check the changelog reflects the mutation, storing the previous state.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": ["yellow"],
            "alpha": [0.3],
        },
        scene_id=0,
    )

    assert space.cuboid_visual_metadata["facecolor"][0] is None
    assert space.cuboid_visual_metadata["alpha"][0] == 0.0
    assert space.cuboid_visual_metadata["linewidth"][0] == 0.5

    assert list(space.composite_index.items()) == [0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]


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

    face_colors = [None, None, "red", "orange"]
    alphas = [0.3, 0, 0.5, 0.6]
    linewidths = [0.5, 0.1, 0.7, 0.8]

    for i in range(4):
        assert space.cuboid_visual_metadata["facecolor"][i] == face_colors[i]
        assert space.cuboid_visual_metadata["alpha"][i] == alphas[i]
        assert space.cuboid_visual_metadata["linewidth"][i] == linewidths[i]

    space.mutate_by_scene(scene=0, facecolor="black", alpha=0.9, linewidth=0.1)

    # Check the changelog reflects the mutation, storing the previous state.
    # TODO: Fix the issue of unintuitive ordering in the mutation subject,
    # currently values are inserted primitives-first.
    assert space.changelog[-1] == bb.Mutation(
        subject={
            "facecolor": [None, None, "red"],
            "alpha": [0.0, 0.3, 0.5],
            "linewidth": [0.1, 0.5, 0.7],
        },
        scene_id=0,
    )

    for i in range(3):
        assert space.cuboid_visual_metadata["facecolor"][i] == "black"
        assert space.cuboid_visual_metadata["alpha"][i] == 0.9
        assert space.cuboid_visual_metadata["linewidth"][i] == 0.1

    assert space.cuboid_visual_metadata["facecolor"][3]
    assert space.cuboid_visual_metadata["alpha"][3] == 0.6
    assert space.cuboid_visual_metadata["linewidth"][3] == 0.8

    assert list(space.cuboid_index.items()) == [1, 1]
    assert space.cuboid_index.get_items_by_timestep(1) == [1]
    assert space.cuboid_index.get_items_by_timestep(4) == [1]
    assert space.cuboid_index.get_items_by_scene(0) == [1]
    assert space.cuboid_index.get_items_by_scene(1) == [1]

    assert list(space.composite_index.items()) == [0, 2, 3, 0, 2]
    assert space.composite_index.get_items_by_timestep(4) == [0, 2]
    assert space.composite_index.get_items_by_scene(0) == [0, 2]
    assert space.composite_index.get_items_by_scene(1) == [3, 0, 2]


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

    # Check the changelog reflects the mutations, storing previous states.
    assert space.changelog[-3:] == [
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
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["composite"],
            creation_type="manual",
            object_names={0: "input-tensor"},
        ),
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
    ]


def test_space_clones_cuboid_from_offset_with_selections() -> None:
    space = bb.Space()

    base_point = np.array([0, 0, 0]).reshape((1, 3))
    space.add_cube(bb.Cube(base_vector=base_point, scale=2.0, name="my-cube"))

    first_offset = np.array([12, 0, 0]).reshape((1, 3))
    second_offset = np.array([0, 12, 0]).reshape((1, 3))
    third_offset = np.array([0, 0, 12]).reshape((1, 3))
    fourth_offset = np.array([32, 0, 0]).reshape((1, 3))

    space.clone_by_offset(first_offset, coordinate=base_point)
    space.clone_by_offset(second_offset, name="my-cube")
    space.clone_by_offset(third_offset, timestep=0)
    space.snapshot()
    space.clone_by_offset(fourth_offset, scene=0)

    expected_mean = np.array([[20], [4], [4]])
    assert np.array_equal(space.mean, expected_mean)
    # TODO: Remove the total field as it's not yet needed (+ probably won't be).
    expected_total = np.array([[160], [32], [32]])
    assert np.array_equal(space.total, expected_total)
    assert space.num_objs == 8
    assert space.object_counter == 8
    assert space.time_step == 5
    assert space.scene_counter == 1

    empty_entries = 2
    expected_point = base_point

    expected_shape = np.array([2, 2, 2]).reshape((1, 3))

    assert np.array_equal(
        space.base_coordinates,
        np.concatenate(
            (
                expected_point,
                expected_point + first_offset,
                expected_point + second_offset,
                expected_point + third_offset,
                expected_point + fourth_offset,
                expected_point + first_offset + fourth_offset,
                expected_point + second_offset + fourth_offset,
                expected_point + third_offset + fourth_offset,
                np.zeros((empty_entries, 3)),
            )
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(expected_shape, (10 - empty_entries, 3)),
                np.zeros((empty_entries, 3)),
            ),
            axis=0,
        ),
    )

    N = space.object_counter
    assert space.cuboid_visual_metadata == {
        "facecolor": [None] * N,
        "linewidth": [0.1] * N,
        "edgecolor": ["black"] * N,
        "alpha": [0.0] * N,
    }

    def populate_addition(count, creation_type, names=None):
        return bb.Addition(
            inserted_count=count,
            object_types_inserted=["primitive"],
            creation_type=creation_type,
            object_names=names,
        )

    assert space.changelog == [
        populate_addition(1, "manual", {0: "my-cube"}),
        populate_addition(1, "ref"),
        populate_addition(1, "ref"),
        populate_addition(1, "ref"),
        populate_addition(4, "ref"),
    ]
    assert list(space.cuboid_index.items()) == [i for i in range(8)]
    assert space.cuboid_index.get_items_by_timestep(4) == [4, 5, 6, 7]


def test_space_clones_composites_from_offset_with_selections() -> None:
    space = bb.Space()

    base_point = np.array([0, 0, 0]).reshape((1, 3))
    w, h, d = 3, 4, 2
    space.add_composite(
        bb.CompositeCube(
            base_vector=base_point, w=w, h=h, d=d, name="my-composite"
        )
    )

    first_offset = np.array([12, 0, 0]).reshape((1, 3))
    second_offset = np.array([0, 12, 0]).reshape((1, 3))
    third_offset = np.array([0, 0, 12]).reshape((1, 3))
    fourth_offset = np.array([32, 0, 0]).reshape((1, 3))

    space.clone_by_offset(first_offset, coordinate=base_point)
    # This should be treated as a no-op.
    space.clone_by_offset(first_offset, coordinate=base_point + 37)
    space.clone_by_offset(second_offset, name="my-composite")
    space.clone_by_offset(third_offset, timestep=0)
    space.snapshot()
    space.clone_by_offset(fourth_offset, scene=0)

    expected_mean = np.array([[20.5], [5], [4]])
    assert np.array_equal(space.mean, expected_mean)
    # TODO: Remove the total field as it's not yet needed (+ probably won't be).
    expected_total = np.array([[164], [40], [32]])
    assert np.array_equal(space.total, expected_total)
    assert space.num_objs == 8
    assert space.object_counter == 8
    assert space.time_step == 5
    assert space.scene_counter == 1

    empty_entries = 2
    expected_point = base_point
    expected_shape = np.array([3, 4, 2]).reshape((1, 3))

    assert np.array_equal(
        space.base_coordinates,
        np.concatenate(
            (
                expected_point,
                expected_point + first_offset,
                expected_point + second_offset,
                expected_point + third_offset,
                expected_point + fourth_offset,
                expected_point + first_offset + fourth_offset,
                expected_point + second_offset + fourth_offset,
                expected_point + third_offset + fourth_offset,
                np.zeros((empty_entries, 3)),
            )
        ),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (
                np.broadcast_to(expected_shape, (10 - empty_entries, 3)),
                np.zeros((empty_entries, 3)),
            ),
            axis=0,
        ),
    )
    N = space.object_counter
    assert space.cuboid_visual_metadata == {
        "facecolor": [None] * N,
        "linewidth": [0.1] * N,
        "edgecolor": ["black"] * N,
        "alpha": [0.0] * N,
    }

    def populate_addition(count, creation_type, names=None):
        return bb.Addition(
            inserted_count=count,
            object_types_inserted=["composite"],
            creation_type=creation_type,
            object_names=names,
        )

    assert space.changelog == [
        populate_addition(1, "manual", {0: "my-composite"}),
        populate_addition(1, "ref"),
        populate_addition(1, "ref"),
        populate_addition(1, "ref"),
        populate_addition(4, "ref"),
    ]
    assert list(space.composite_index.items()) == [i for i in range(8)]
    assert space.composite_index.get_items_by_timestep(4) == [4, 5, 6, 7]


def test_space_clone_from_offset_only_uses_one_selection() -> None:
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


def test_space_clones_composites_from_offset_with_updated_visuals() -> None:
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
    new_face_colors = ["black", "brown", "green", "blue"]
    new_edge_colors = ["white", "black", "purple", "white"]
    all_face_colors = [
        primitive_fc
        for fc in [None, "blue", None, None] + new_face_colors
        for primitive_fc in [fc]
    ]
    all_edge_colors = [
        primitive_ec
        for ec in ["black", "white", "black", "black"] + new_edge_colors
        for primitive_ec in [ec]
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
        "linewidth": [0.1] * 8,
        "edgecolor": all_edge_colors,
        "alpha": [0.0, 0.2, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    }


def test_space_transforms_primitive_by_coordinate() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    shifted_point = point + translate
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    reflected_point = shifted_point * reflect
    scale = np.array([2, 2, 2])
    space.transform_by_coordinate(coordinate=point, translate=translate)
    space.transform_by_coordinate(coordinate=shifted_point, reflect=reflect)
    space.transform_by_coordinate(coordinate=reflected_point, scale=scale)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate, transform_name="translation", coordinate=point
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            coordinate=shifted_point,
        ),
        bb.Transform(
            transform=1 / scale,
            transform_name="scale",
            coordinate=reflected_point,
        ),
    ]

    expected_point = (point + translate) * reflect * scale
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[2, -2, -2]])

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.cuboid_index.items()) == [0, 0, 0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]
    assert space.cuboid_index.get_items_by_timestep(2) == [0]
    assert space.cuboid_index.get_items_by_timestep(3) == [0]


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

    translate = np.array([3, 3, 3])
    shifted_point = point + translate
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    space.transform_by_coordinate(coordinate=point, translate=translate)
    space.transform_by_coordinate(coordinate=shifted_point, reflect=reflect)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            coordinate=point,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            coordinate=shifted_point,
        ),
    ]

    expected_point = (point + translate) * reflect
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[w, h, d]]) * reflect

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.composite_index.items()) == [0, 0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]
    assert space.composite_index.get_items_by_timestep(2) == [0]


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

    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    shifted_point = point + translate
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    space.transform_by_coordinate(coordinate=point, translate=translate)
    space.transform_by_coordinate(coordinate=shifted_point, reflect=reflect)

    # Check the changelog reflects the transforms, storing previous states§.
    assert space.changelog[2:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            coordinate=point,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            coordinate=shifted_point,
        ),
    ]

    expected_point = (point + translate) * reflect
    expected_point = expected_point.reshape((1, 3))

    empty_entries = 8
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate(
            (expected_point, expected_point, np.zeros((empty_entries, 3))),
            axis=0,
        ),
    )

    composite_shape = np.array([[w, -h, -d]])
    cube_shape = np.array([[1, -1, -1]])
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate(
            (composite_shape, cube_shape, np.zeros((empty_entries, 3))),
            axis=0,
        ),
    )

    assert list(space.cuboid_index.items()) == [1, 1, 1]
    assert list(space.composite_index.items()) == [0, 0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [1]
    assert space.cuboid_index.get_items_by_timestep(2) == [1]
    assert space.composite_index.get_items_by_timestep(2) == [0]
    assert space.cuboid_index.get_items_by_timestep(3) == [1]
    assert space.composite_index.get_items_by_timestep(3) == [0]


def test_space_transforms_primitive_by_name() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point, name="my-primitive"))

    translate = np.array([3, 3, 3])
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    scale = np.array([2, 2, 2])
    space.transform_by_name(name="my-primitive", translate=translate)
    space.transform_by_name(name="my-primitive", reflect=reflect)
    space.transform_by_name(name="my-primitive", scale=scale)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            name="my-primitive",
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            name="my-primitive",
        ),
        bb.Transform(
            transform=1 / scale,
            transform_name="scale",
            name="my-primitive",
        ),
    ]

    expected_point = (point + translate) * reflect * scale
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[2, -2, -2]])

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.cuboid_index.items()) == [0, 0, 0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]
    assert space.cuboid_index.get_items_by_timestep(2) == [0]
    assert space.cuboid_index.get_items_by_timestep(3) == [0]


def test_space_transforms_composite_by_name() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(
        bb.CompositeCube(base_vector=point, w=w, h=h, d=d, name="my-composite")
    )

    translate = np.array([3, 3, 3])
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    space.transform_by_name(name="my-composite", translate=translate)
    space.transform_by_name(name="my-composite", reflect=reflect)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            name="my-composite",
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            name="my-composite",
        ),
    ]

    expected_point = (point + translate) * reflect
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[w, h, d]]) * reflect

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.composite_index.items()) == [0, 0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]
    assert space.composite_index.get_items_by_timestep(2) == [0]


def test_space_transforms_primitive_by_timestep_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    scale = np.array([2, 2, 2])
    space.transform_by_timestep(timestep=0, translate=translate)
    # Possibly counter-intuitive, but these are equally valid as timestep==1.
    space.transform_by_timestep(timestep=0, reflect=reflect)
    space.transform_by_timestep(timestep=0, scale=scale)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            timestep_id=0,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            timestep_id=0,
        ),
        bb.Transform(
            transform=1 / scale,
            transform_name="scale",
            timestep_id=0,
        ),
    ]

    expected_point = (point + translate) * reflect * scale
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[2, -2, -2]])

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.cuboid_index.items()) == [0, 0, 0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]
    assert space.cuboid_index.get_items_by_timestep(2) == [0]
    assert space.cuboid_index.get_items_by_timestep(3) == [0]


def test_space_transforms_composite_by_timestep_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(bb.CompositeCube(base_vector=point, w=w, h=h, d=d))

    translate = np.array([3, 3, 3])
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    space.transform_by_timestep(timestep=0, translate=translate)
    # Possibly counter-intuitive, but this is equally valid as timestep==1.
    space.transform_by_timestep(timestep=0, reflect=reflect)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            timestep_id=0,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            timestep_id=0,
        ),
    ]

    expected_point = (point + translate) * reflect
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[w, h, d]]) * reflect

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.composite_index.items()) == [0, 0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]
    assert space.composite_index.get_items_by_timestep(2) == [0]


def test_space_transforms_primitive_by_scene_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([3, 3, 3])
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    scale = np.array([2, 2, 2])
    space.transform_by_scene(scene=0, translate=translate)
    space.transform_by_scene(scene=0, reflect=reflect)
    space.transform_by_scene(scene=0, scale=scale)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            scene_id=0,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            scene_id=0,
        ),
        bb.Transform(
            transform=1 / scale,
            transform_name="scale",
            scene_id=0,
        ),
    ]

    expected_point = (point + translate) * reflect * scale
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[2, -2, -2]])

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.cuboid_index.items()) == [0, 0, 0, 0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]
    assert space.cuboid_index.get_items_by_timestep(1) == [0]
    assert space.cuboid_index.get_items_by_timestep(2) == [0]
    assert space.cuboid_index.get_items_by_timestep(3) == [0]


def test_space_transforms_composite_by_scene_id() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3])
    w, h, d = 4, 3, 2
    space.add_composite(bb.CompositeCube(base_vector=point, w=w, h=h, d=d))

    translate = np.array([3, 3, 3])
    # TODO: Make reflection slightly more readable/interpretable.
    # This reflects about the x-axis.
    reflect = np.array([1, -1, -1])
    space.transform_by_scene(scene=0, translate=translate)
    space.transform_by_scene(scene=0, reflect=reflect)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[1:] == [
        bb.Transform(
            transform=-translate,
            transform_name="translation",
            scene_id=0,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            scene_id=0,
        ),
    ]

    expected_point = (point + translate) * reflect
    expected_point = expected_point.reshape((1, 3))
    expected_shape = np.array([[w, h, d]]) * reflect

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.composite_index.items()) == [0, 0, 0]
    assert space.composite_index.get_items_by_timestep(0) == [0]
    assert space.composite_index.get_items_by_timestep(1) == [0]
    assert space.composite_index.get_items_by_timestep(2) == [0]


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

    space.add_cube(bb.Cube(base_vector=np.array([12, 14, 3]), facecolor="pink"))

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
    third_translate = np.array([3, 2, 1])
    reflect = np.array([1, -1, -1])
    scale = np.array([2, 2, 2])
    space.transform_by_scene(scene=0, translate=first_translate)
    space.transform_by_scene(scene=1, translate=second_translate)
    space.transform_by_name(name="input-tensor", translate=third_translate)
    space.transform_by_timestep(timestep=1, scale=scale)
    coordinates_before = np.copy(space.base_coordinates)
    # Having two reflections should lead to the identity.
    space.transform_by_scene(scene=1, reflect=reflect)
    space.transform_by_scene(scene=1, reflect=reflect)
    coordinates_after = space.base_coordinates
    assert np.array_equal(coordinates_before, coordinates_after)

    # Check the changelog reflects the transforms, storing previous states.
    assert space.changelog[4:] == [
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
        bb.Transform(
            transform=1 / scale,
            transform_name="scale",
            timestep_id=1,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            scene_id=1,
        ),
        bb.Transform(
            transform=reflect,
            transform_name="reflection",
            scene_id=1,
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
    reflect = np.array([1, -1, -1])

    expected_name_err_msg = "The provided name does not exist in this space."
    with pytest.raises(Exception, match=expected_name_err_msg):
        space.transform_by_name("not-a-valid-name", translate=translate)
    with pytest.raises(Exception, match=expected_name_err_msg):
        space.transform_by_name("not-a-valid-name", reflect=reflect)

    expected_timestep_err_msg = (
        "The provided timestep is invalid in this space."
    )
    with pytest.raises(Exception, match=expected_timestep_err_msg):
        space.transform_by_timestep(timestep=3, translate=translate)
    with pytest.raises(Exception, match=expected_timestep_err_msg):
        space.transform_by_timestep(timestep=3, reflect=reflect)

    # The scene is technically valid (it is the current scene), but it is empty,
    # so it silently does nothing.
    space.transform_by_scene(scene=1, translate=translate)
    space.transform_by_scene(scene=1, reflect=reflect)

    # An error isn't meaningful in this case, so it silently does nothing.
    space.transform_by_coordinate(np.array([64, 32, 16]), translate=translate)
    space.transform_by_coordinate(np.array([64, 32, 16]), reflect=reflect)

    # None of the above transforms have any effect and are not reflected in the
    # history.
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["composite"],
            creation_type="manual",
            object_names={0: "input-tensor"},
        ),
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        ),
    ]


def test_space_transforms_nothing_with_trivial_transform() -> None:
    space = bb.Space()

    point = np.array([1, 2, 3]).reshape((1, 3))
    space.add_cube(bb.Cube(base_vector=point))

    translate = np.array([0, 0, 0])
    # TODO: Make reflection slightly more readable/interpretable.
    reflect = np.array([1, 1, 1])
    space.transform_by_scene(scene=0, translate=translate)
    space.transform_by_scene(scene=0, reflect=reflect)

    # Check the changelog reflects no transforms.
    assert space.changelog == [
        bb.Addition(
            inserted_count=1,
            object_types_inserted=["primitive"],
            creation_type="manual",
            object_names=None,
        )
    ]

    expected_point = point
    expected_shape = np.array([1, 1, 1]).reshape((1, 3))

    empty_entries = 9
    assert np.array_equal(
        space.base_coordinates,
        np.concatenate((expected_point, np.zeros((empty_entries, 3))), axis=0),
    )
    assert np.array_equal(
        space.cuboid_shapes,
        np.concatenate((expected_shape, np.zeros((empty_entries, 3))), axis=0),
    )

    assert list(space.cuboid_index.items()) == [0]
    assert space.cuboid_index.get_items_by_timestep(0) == [0]


def test_space_scale_does_not_apply_to_composites() -> None:
    space = bb.Space()

    space.add_composite(
        bb.CompositeCube(base_vector=np.array([1, 2, 3]), w=4, h=3, d=2)
    )

    s = np.array([2, 2, 2])
    expected_err_msg = "Scale may only be applied to primitives."
    with pytest.raises(ValueError, match=expected_err_msg):
        space.transform_by_coordinate(coordinate=np.array([1, 2, 3]), scale=s)


def test_space_scale_cannot_be_non_positive() -> None:
    space = bb.Space()

    space.add_cube(bb.Cube(base_vector=np.array([1, 2, 3]), scale=2))

    s = np.array([2, -2, -2])
    expected_err_msg = "Scale may only contain positive values."
    with pytest.raises(ValueError, match=expected_err_msg):
        space.transform_by_coordinate(coordinate=np.array([1, 2, 3]), scale=s)


def test_space_supports_composites_with_classic_style() -> None:
    ...


def test_space_correctly_reorients_data() -> None:
    ...
