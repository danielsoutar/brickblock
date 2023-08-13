import brickblock as bb


def mock_arbitrary_objects() -> list[int | slice]:
    return [
        0,
        1,
        2,
        3,
        slice(4, 31),
        slice(31, 100),
        100,
        slice(101, 300),
        slice(300, 600),
        slice(600, 1000),
        1000,
        1001,
        1002,
        slice(1003, 9000),
    ]


def test_index_defaults() -> None:
    index = bb.SpaceIndex()
    assert index._primitive_buffer == []
    assert index._primitive_timestep_index == []
    assert index._primitive_scene_index == []
    assert index._composite_buffer == []
    assert index._composite_timestep_index == []
    assert index._composite_scene_index == []


def test_index_correctly_populated_after_one_primitive() -> None:
    index = bb.SpaceIndex()
    index.add_primitive_to_index(0, 0, 0)
    assert index._primitive_buffer == [0]
    assert index._primitive_timestep_index == [1]
    assert index._primitive_scene_index == [1]
    assert index._composite_buffer == []
    assert index._composite_timestep_index == []
    assert index._composite_scene_index == []


def test_index_correctly_populated_after_many_primitives() -> None:
    index = bb.SpaceIndex()
    index.add_primitive_to_index(0, 0, 0)
    index.add_primitive_to_index(1, 1, 0)
    index.add_primitive_to_index(2, 2, 0)
    assert index._primitive_buffer == [0, 1, 2]
    assert index._primitive_timestep_index == [1, 2, 3]
    assert index._primitive_scene_index == [3]
    assert index._composite_buffer == []
    assert index._composite_timestep_index == []
    assert index._composite_scene_index == []


def test_index_correctly_populated_after_one_composite() -> None:
    index = bb.SpaceIndex()
    index.add_composite_to_index(slice(0, 1), 0, 0)
    assert index._primitive_buffer == []
    assert index._primitive_timestep_index == []
    assert index._primitive_scene_index == []
    assert index._composite_buffer == [slice(0, 1)]
    assert index._composite_timestep_index == [1]
    assert index._composite_scene_index == [1]


def test_index_correctly_populated_after_many_composites() -> None:
    index = bb.SpaceIndex()
    index.add_composite_to_index(slice(0, 1), 0, 0)
    index.add_composite_to_index(slice(1, 5), 1, 0)
    index.add_composite_to_index(slice(5, 12), 2, 0)
    assert index._primitive_buffer == []
    assert index._primitive_timestep_index == []
    assert index._primitive_scene_index == []
    assert index._composite_buffer == [slice(0, 1), slice(1, 5), slice(5, 12)]
    assert index._composite_timestep_index == [1, 2, 3]
    assert index._composite_scene_index == [3]


def test_index_correctly_populated_with_primitive_and_composite() -> None:
    index = bb.SpaceIndex()
    index.add_primitive_to_index(0, timestep_id=0, scene_id=0)
    index.add_composite_to_index(slice(1, 2), timestep_id=1, scene_id=0)
    assert index._primitive_buffer == [0]
    assert index._primitive_timestep_index == [1]
    assert index._primitive_scene_index == [1]
    assert index._composite_buffer == [slice(1, 2)]
    assert index._composite_timestep_index == [0, 1]
    assert index._composite_scene_index == [1]


def test_index_correctly_populated_with_primitives_then_composite() -> None:
    index = bb.SpaceIndex()
    index.add_primitive_to_index(0, timestep_id=0, scene_id=0)
    index.add_primitive_to_index(1, timestep_id=1, scene_id=0)
    index.add_primitive_to_index(2, timestep_id=2, scene_id=0)
    index.add_primitive_to_index(3, timestep_id=3, scene_id=0)
    index.add_composite_to_index(slice(4, 200), timestep_id=4, scene_id=0)
    assert index._primitive_buffer == [0, 1, 2, 3]
    assert index._primitive_timestep_index == [1, 2, 3, 4]
    assert index._primitive_scene_index == [4]
    assert index._composite_buffer == [slice(4, 200)]
    assert index._composite_timestep_index == [0, 0, 0, 0, 1]
    assert index._composite_scene_index == [1]


def test_index_correctly_populated_with_primitives_then_composites() -> None:
    index = bb.SpaceIndex()
    index.add_primitive_to_index(0, timestep_id=0, scene_id=0)
    index.add_primitive_to_index(1, timestep_id=1, scene_id=0)
    index.add_primitive_to_index(2, timestep_id=2, scene_id=0)
    index.add_primitive_to_index(3, timestep_id=3, scene_id=0)
    index.add_composite_to_index(slice(4, 200), timestep_id=4, scene_id=0)
    index.add_composite_to_index(slice(200, 405), timestep_id=5, scene_id=0)
    index.add_composite_to_index(slice(405, 610), timestep_id=6, scene_id=0)
    index.add_composite_to_index(slice(610, 1200), timestep_id=7, scene_id=0)
    assert index._primitive_buffer == [0, 1, 2, 3]
    assert index._primitive_timestep_index == [1, 2, 3, 4]
    assert index._primitive_scene_index == [4]
    assert index._composite_buffer == [
        slice(4, 200),
        slice(200, 405),
        slice(405, 610),
        slice(610, 1200),
    ]
    assert index._composite_timestep_index == [0, 0, 0, 0, 1, 2, 3, 4]
    assert index._composite_scene_index == [4]


def test_index_correctly_populated_with_composites_then_primitive() -> None:
    index = bb.SpaceIndex()
    index.add_composite_to_index(slice(0, 2000), timestep_id=0, scene_id=0)
    index.add_composite_to_index(slice(2000, 3000), timestep_id=1, scene_id=0)
    index.add_composite_to_index(slice(3000, 3500), timestep_id=2, scene_id=0)
    index.add_composite_to_index(slice(3500, 3750), timestep_id=3, scene_id=0)
    index.add_primitive_to_index(3750, timestep_id=4, scene_id=0)
    assert index._primitive_buffer == [3750]
    assert index._primitive_timestep_index == [0, 0, 0, 0, 1]
    assert index._primitive_scene_index == [1]
    assert index._composite_buffer == [
        slice(0, 2000),
        slice(2000, 3000),
        slice(3000, 3500),
        slice(3500, 3750),
    ]
    assert index._composite_timestep_index == [1, 2, 3, 4]
    assert index._composite_scene_index == [4]


def test_index_correctly_populated_with_composites_then_primitives() -> None:
    index = bb.SpaceIndex()
    index.add_composite_to_index(slice(0, 2000), timestep_id=0, scene_id=0)
    index.add_composite_to_index(slice(2000, 3000), timestep_id=1, scene_id=0)
    index.add_composite_to_index(slice(3000, 3500), timestep_id=2, scene_id=0)
    index.add_composite_to_index(slice(3500, 3750), timestep_id=3, scene_id=0)
    index.add_primitive_to_index(3750, timestep_id=4, scene_id=0)
    index.add_primitive_to_index(3751, timestep_id=5, scene_id=0)
    index.add_primitive_to_index(3752, timestep_id=6, scene_id=0)
    index.add_primitive_to_index(3753, timestep_id=7, scene_id=0)
    assert index._primitive_buffer == [3750, 3751, 3752, 3753]
    assert index._primitive_timestep_index == [0, 0, 0, 0, 1, 2, 3, 4]
    assert index._primitive_scene_index == [4]
    assert index._composite_buffer == [
        slice(0, 2000),
        slice(2000, 3000),
        slice(3000, 3500),
        slice(3500, 3750),
    ]
    assert index._composite_timestep_index == [1, 2, 3, 4]
    assert index._composite_scene_index == [4]


def test_index_correctly_populated_with_interleaved_objects() -> None:
    index = bb.SpaceIndex()
    index.add_primitive_to_index(0, timestep_id=0, scene_id=0)
    index.add_composite_to_index(slice(1, 28), timestep_id=1, scene_id=0)
    index.add_primitive_to_index(28, timestep_id=2, scene_id=0)
    index.add_composite_to_index(slice(29, 404), timestep_id=3, scene_id=0)
    index.add_primitive_to_index(404, timestep_id=4, scene_id=0)
    index.add_composite_to_index(slice(405, 900), timestep_id=5, scene_id=0)
    index.add_primitive_to_index(900, timestep_id=6, scene_id=0)
    index.add_composite_to_index(slice(900, 9000), timestep_id=7, scene_id=0)
    assert index._primitive_buffer == [0, 28, 404, 900]
    assert index._primitive_timestep_index == [1, 1, 2, 2, 3, 3, 4]
    assert index._primitive_scene_index == [4]
    assert index._composite_buffer == [
        slice(1, 28),
        slice(29, 404),
        slice(405, 900),
        slice(900, 9000),
    ]
    assert index._composite_timestep_index == [0, 1, 1, 2, 2, 3, 3, 4]
    assert index._composite_scene_index == [4]


def test_index_correctly_populated_with_arbitrary_objects() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    expected_primitive_buffer = [0, 1, 2, 3, 100, 1000, 1001, 1002]
    expected_primitive_timestep_index = [1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 7, 8]
    expected_primitive_scene_index = [5, 8]
    assert index._primitive_buffer == expected_primitive_buffer
    assert index._primitive_timestep_index == expected_primitive_timestep_index
    assert index._primitive_scene_index == expected_primitive_scene_index
    expected_composite_buffer = [
        slice(4, 31),
        slice(31, 100),
        slice(101, 300),
        slice(300, 600),
        slice(600, 1000),
        slice(1003, 9000),
    ]
    expected_composite_timestep_index = [0] * 4 + [1, 2, 2, 3, 4, 5, 5, 5, 5, 6]
    expected_composite_scene_index = [2, 6]
    assert index._composite_buffer == expected_composite_buffer
    assert index._composite_timestep_index == expected_composite_timestep_index
    assert index._composite_scene_index == expected_composite_scene_index


def test_current_scene_is_valid() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    expected_number_of_scenes = 2
    assert index.current_scene_is_valid(expected_number_of_scenes)
    assert not index.current_scene_is_valid(expected_number_of_scenes - 1)
    assert not index.current_scene_is_valid(expected_number_of_scenes + 1)


def test_index_can_return_object_iterators_in_order_of_insertion() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    primitives_iterator = index.primitives()

    assert [0, 1, 2, 3, 100, 1000, 1001, 1002] == list(primitives_iterator)

    composites_iterator = index.composites()

    expected_composites = [
        slice(4, 31),
        slice(31, 100),
        slice(101, 300),
        slice(300, 600),
        slice(600, 1000),
        slice(1003, 9000),
    ]
    assert expected_composites == list(composites_iterator)


def test_index_can_get_primitives_by_timestep() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    expected_results = [[0], [1], [2], [3], [100], [1000], [1001], [1002]]
    timesteps_to_evaluate = [0, 1, 2, 3, 6, 10, 11, 12]

    for expected, timestep in zip(expected_results, timesteps_to_evaluate):
        assert expected == index.get_primitives_by_timestep(timestep)

    # Check for empty result with a valid timestep with no primitives.
    assert [] == index.get_primitives_by_timestep(4)
    # Check for empty result with an invalid timestep.
    assert [] == index.get_primitives_by_timestep(13)


def test_index_can_get_composites_by_timestep() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    expected_results = [
        [slice(4, 31)],
        [slice(31, 100)],
        [slice(101, 300)],
        [slice(300, 600)],
        [slice(600, 1000)],
        [slice(1003, 9000)],
    ]
    timesteps_to_evaluate = [4, 5, 7, 8, 9, 13]

    for expected, timestep in zip(expected_results, timesteps_to_evaluate):
        assert expected == index.get_composites_by_timestep(timestep)

    # Check for empty result with a valid timestep with no composites.
    assert [] == index.get_composites_by_timestep(3)
    # Check for empty result with an invalid timestep.
    assert [] == index.get_composites_by_timestep(15)


def test_index_can_get_primitives_by_scene() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    index.add_composite_to_index(slice(9000, 9999), timestep_id=14, scene_id=2)

    expected_results = [[0, 1, 2, 3, 100], [1000, 1001, 1002]]
    scenes_to_evaluate = [0, 1]

    for expected, scene in zip(expected_results, scenes_to_evaluate):
        assert expected == index.get_primitives_by_scene(scene)

    # Check for empty result with a valid scene with no primitives.
    assert [] == index.get_primitives_by_scene(2)
    # Check for empty result with an invalid scene.
    assert [] == index.get_primitives_by_scene(10)


def test_index_can_get_composites_by_scene() -> None:
    index = bb.SpaceIndex()

    objects = mock_arbitrary_objects()

    for i, obj in enumerate(objects):
        scene_id = 1 if i >= 7 else 0

        if isinstance(obj, int):
            index.add_primitive_to_index(obj, timestep_id=i, scene_id=scene_id)
        else:
            index.add_composite_to_index(obj, timestep_id=i, scene_id=scene_id)

    index.add_primitive_to_index(9000, timestep_id=14, scene_id=2)

    expected_results = [
        [slice(4, 31), slice(31, 100)],
        [
            slice(101, 300),
            slice(300, 600),
            slice(600, 1000),
            slice(1003, 9000),
        ],
    ]
    scenes_to_evaluate = [0, 1]

    for expected, scene in zip(expected_results, scenes_to_evaluate):
        assert expected == index.get_composites_by_scene(scene)

    # Check for empty result with a valid scene with no composites.
    assert [] == index.get_composites_by_scene(2)
    # Check for empty result with an invalid scene.
    assert [] == index.get_composites_by_scene(10)
