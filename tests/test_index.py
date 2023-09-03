import pytest

import brickblock as bb


def mock_arbitrary_objects() -> list[int]:
    return [i for i in range(14)]


def test_index_defaults() -> None:
    index = bb.TemporalIndex()
    assert index._item_buffer == []
    assert index._item_timestep_index == []
    assert index._item_scene_index == []


def test_index_correctly_populated_after_one_item() -> None:
    index = bb.TemporalIndex()
    index.add_item_to_index(0, 0, 0)
    assert index._item_buffer == [0]
    assert index._item_timestep_index == [1]
    assert index._item_scene_index == [1]


def test_index_correctly_populated_after_many_items() -> None:
    index = bb.TemporalIndex()
    index.add_item_to_index(0, 0, 0)
    index.add_item_to_index(1, 1, 0)
    index.add_item_to_index(2, 2, 0)
    assert index._item_buffer == [0, 1, 2]
    assert index._item_timestep_index == [1, 2, 3]
    assert index._item_scene_index == [3]


def test_current_scene_is_valid() -> None:
    index = bb.TemporalIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0
        index.add_item_to_index(obj, timestep_id=i, scene_id=scene_id)

    expected_number_of_scenes = 2
    assert index.current_scene_is_valid(expected_number_of_scenes)
    assert not index.current_scene_is_valid(expected_number_of_scenes - 1)
    assert not index.current_scene_is_valid(expected_number_of_scenes + 1)


def test_index_can_return_object_iterators_in_order_of_insertion() -> None:
    index = bb.TemporalIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0
        index.add_item_to_index(obj, timestep_id=i, scene_id=scene_id)

    assert [i for i in range(len(objects))] == list(index.items())


def test_index_can_get_items_by_timestep() -> None:
    index = bb.TemporalIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0
        index.add_item_to_index(obj, timestep_id=i, scene_id=scene_id)

    expected_results = [[i] for i in range(len(objects))]
    timesteps_to_evaluate = [i for i in range(len(objects))]

    for expected, timestep in zip(expected_results, timesteps_to_evaluate):
        assert expected == index.get_items_by_timestep(timestep)


def test_index_gets_no_items_by_timestep_when_timestep_is_empty() -> None:
    index = bb.TemporalIndex()

    objects = [0, 1, 2, 3, 4, 5, 6]
    timesteps = [0, 1, 2, 3, 19, 20, 21]

    for t, obj in zip(timesteps, objects):
        scene_id = 1 if t >= 7 else 0
        index.add_item_to_index(obj, timestep_id=t, scene_id=scene_id)

    # Check for empty result with a valid timestep with no items.
    assert [] == index.get_items_by_timestep(4)
    # Check for empty result with an invalid timestep.
    assert [] == index.get_items_by_timestep(13)


def test_index_can_get_items_by_scene() -> None:
    index = bb.TemporalIndex()

    objects = mock_arbitrary_objects()
    timesteps = [0, 1, 2, 3] + [19 + i for i in range(len(objects) - 4)]

    for t, obj in zip(timesteps, objects):
        scene_id = 1 if t >= 7 else 0
        index.add_item_to_index(obj, timestep_id=t, scene_id=scene_id)

    expected_results = [
        [i for i in range(4)],
        [i for i in range(4, len(objects))],
    ]
    scenes_to_evaluate = [0, 1]

    for expected, scene in zip(expected_results, scenes_to_evaluate):
        assert expected == index.get_items_by_scene(scene)

    # Check for empty result with a valid scene with no items.
    assert [] == index.get_items_by_scene(2)


def test_index_clears_items_by_latest_timestep() -> None:
    index = bb.TemporalIndex()

    objects = mock_arbitrary_objects()
    timesteps = [0, 1, 2, 3] + [19 + i for i in range(len(objects) - 4)]

    for t, obj in zip(timesteps, objects):
        scene_id = 1 if t >= 7 else 0
        index.add_item_to_index(obj, timestep_id=t, scene_id=scene_id)

    # Although only 14 items were added into the buffer, the timestep indices
    # used indicate that the last 10 values were 15 timesteps into the future.
    # This gives 29 elements, so the last timestep is index 28.
    popped_item = index.clear_items_in_latest_timestep(28)

    assert popped_item == [13]
    assert index._item_buffer == [i for i in range(13)]
    assert index.get_items_by_scene(1) == [i for i in range(4, 13)]

    # Check that popping until hitting the first scene is supported.
    id = 27
    while len(index.get_items_by_scene(1)) > 0:
        popped_item = index.clear_items_in_latest_timestep(id)

        assert popped_item == [id - 15]
        assert index._item_buffer == [i for i in range(id - 15)]
        assert index.get_items_by_scene(1) == [i for i in range(4, id - 15)]

        id -= 1

    assert index._item_timestep_index == [1, 2, 3, 4] + [4] * 15
    assert index._item_scene_index == [4]

    # Then check that we can fully remove everything and still be valid.

    while id > 3:
        # These are dummy values - timesteps that are valid but no items of
        # the given kind were referenced.
        index.clear_items_in_latest_timestep(id)
        assert popped_item == [4]
        assert index.get_items_by_scene(0) == [0, 1, 2, 3]
        assert index.get_items_by_scene(1) == []

        id -= 1

    index.clear_items_in_latest_timestep(3)
    index.clear_items_in_latest_timestep(2)
    index.clear_items_in_latest_timestep(1)
    popped_item = index.clear_items_in_latest_timestep(0)

    assert popped_item == [0]
    assert index._item_buffer == []
    assert index._item_timestep_index == []
    assert index._item_scene_index == []


def test_index_clears_items_by_latest_timestep_nothing_on_empty_index() -> None:
    index = bb.TemporalIndex()
    popped_item = index.clear_items_in_latest_timestep(0)
    assert popped_item == []


def test_index_clears_items_by_latest_timestep_on_single_item_index() -> None:
    index = bb.TemporalIndex()
    index.add_item_to_index(0, timestep_id=0, scene_id=0)
    popped_item = index.clear_items_in_latest_timestep(0)
    assert popped_item == [0]
    assert index._item_buffer == []
    assert index._item_timestep_index == []
    assert index._item_scene_index == []


def test_index_clears_items_by_latest_timestep_error_on_invalid_timestep() -> (
    None
):
    index = bb.TemporalIndex()

    objects = mock_arbitrary_objects()
    timesteps = [0, 1, 2, 3] + [19 + i for i in range(len(objects) - 4)]

    for t, obj in zip(timesteps, objects):
        scene_id = 1 if t >= 7 else 0
        index.add_item_to_index(obj, timestep_id=t, scene_id=scene_id)

    expected_name_err_msg = (
        "This function only supports removing items for the latest timestep."
    )
    with pytest.raises(ValueError, match=expected_name_err_msg):
        index.clear_items_in_latest_timestep(5)
