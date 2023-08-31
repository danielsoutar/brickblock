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
