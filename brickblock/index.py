from collections.abc import Iterator


class TemporalIndex:
    _item_buffer: list[int]
    _item_timestep_index: list[int]
    _item_scene_index: list[int]

    def __init__(self):
        self._item_buffer = []
        self._item_timestep_index = []
        self._item_scene_index = []

    # TODO: Improve docstring for this function - likely worth adding examples.
    def _add_entry_to_offset_index(self, id: int, index: list[int]) -> None:
        """
        Add an offset for `id` to the given `index`.

        If the given ID is not an immediate successor of the latest entry in the
        index (e.g. no items have been added for several timesteps or
        scenes), then intermediate entries are added with offsets such that they
        produce zero-length sequences when indexing into the relevant buffer.

        # Args
            id: The ID which indicates the number of entries to add to the
                index.
            index: The index to add an entry to, or increment the last value if
                `id` represents the latest entry.
        """
        # If the offset index is already set, increment, otherwise set with
        # the number of entries to offset from.
        num_entries = len(index)
        if num_entries == (id + 1):
            index[-1] += 1
        else:
            is_empty = num_entries == 0
            val = 0 if is_empty else index[-1]
            index.extend([val] * (id - num_entries))
            index.append(val + 1)

    def add_item_to_index(
        self, id: int, timestep_id: int, scene_id: int
    ) -> None:
        """
        Add an `id` to the item buffer, along with relevant timestep and scene
        information.

        This will update the timestep and scene indices accordingly. If the
        given timestep and scene IDs are not immediate successors of the latest
        entries in the buffer (i.e. no items have been added for several
        timesteps or scenes), then the intermediate entries in the
        timestep/scene indices are populated with offsets such that they produce
        zero-length sequences when indexing into the item buffer.

        # Args
            id: The ID to add into the item buffer.
            timestep_id: The ID of the timestep this item is in.
            scene_id: The ID of the scene this item is in.
        """
        self._item_buffer.append(id)

        self._add_entry_to_offset_index(
            id=timestep_id, index=self._item_timestep_index
        )
        self._add_entry_to_offset_index(
            id=scene_id, index=self._item_scene_index
        )

    def current_scene_is_valid(self, expected_num_scenes: int) -> bool:
        """
        Return whether the current scene is valid, i.e. has at least one item
        referenced in the current scene.
        """
        return len(self._item_scene_index) == expected_num_scenes

    def items(self) -> Iterator[int]:
        """
        Get all distinct items in the index.
        """
        for item in self._item_buffer:
            yield item

    def _extract_objects_by_id(self, id: int, index: list[int]) -> slice:
        """
        Retrieve a slice with its limits given by the `id`-th entry in `index`.

        If `id` is outside the limits of the index, or the index is empty,
        return a slice such that using it yields an empty list.

        # Args
            id: The ID to index for.
            index: The object to index into to retrieve the offsets.
        """
        # This accounts for the empty index case and the case where the index
        # does not have the requisite number of entries.
        is_present = len(index) >= (id + 1)
        if not is_present:
            return slice(0, 0)

        is_first = id == 0
        is_empty = len(index) == 0
        start = 0 if is_first else index[id - 1]
        stop = 0 if is_empty else index[id]
        return slice(start, stop)

    def get_items_by_timestep(self, timestep_id: int) -> list[int]:
        """
        Get all items in the index with timestep equal to `timestep_id`, in
        order of insertion.

        If the given timestep has no items, return an empty list.

        # Args
            timestep_id: The ID of the timestep to query over.
        """
        subset = self._extract_objects_by_id(
            id=timestep_id, index=self._item_timestep_index
        )
        return self._item_buffer[subset]

    def get_items_by_scene(self, scene_id: int) -> list[int]:
        """
        Get all items in the index with scene equal to `scene_id`, in order of
        of insertion.

        If the given scene has no items, return an empty list.

        # Args
            scene_id: The ID of the scene to query over.
        """
        subset = self._extract_objects_by_id(
            id=scene_id, index=self._item_scene_index
        )
        return self._item_buffer[subset]
