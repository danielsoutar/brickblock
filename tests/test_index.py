import numpy as np

import brickblock as bb


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


def test_index_correctly_updated_with_interleaved_objects() -> None:
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


def test_current_scene_is_valid() -> None:
    ...
