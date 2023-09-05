from collections.abc import Iterator
from typing import Any


class TemporalIndex:
    _item_buffer: list[Any]
    _item_timestep_index: list[int]
    _item_scene_index: list[int]

    def __init__(self):
        self._item_buffer = []
        self._item_timestep_index = []
        self._item_scene_index = []

    def __eq__(self, __value: object) -> bool:
        return (
            self._item_buffer == __value._item_buffer
            and self._item_timestep_index == __value._item_timestep_index
            and self._item_scene_index == __value._item_scene_index
        )

    def __len__(self) -> int:
        # This returns the length of the *item buffer, not the indices!
        return len(self._item_buffer)

    def __getitem__(self, idx: int) -> Any:
        return self._item_buffer[idx]

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
        self, item: Any, timestep_id: int, scene_id: int
    ) -> None:
        """
        Add an `item` to the item buffer, along with relevant timestep and scene
        information.

        This will update the timestep and scene indices accordingly. If the
        given timestep and scene IDs are not immediate successors of the latest
        entries in the buffer (i.e. no items have been added for several
        timesteps or scenes), then the intermediate entries in the
        timestep/scene indices are populated with offsets such that they produce
        zero-length sequences when indexing into the item buffer.

        # Args
            item: The item to add into the item buffer.
            timestep_id: The ID of the timestep this item is in.
            scene_id: The ID of the scene this item is in.
        """
        self._item_buffer.append(item)

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

    def items(self) -> Iterator[Any]:
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

    def get_items_by_timestep(self, timestep_id: int) -> list[Any]:
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

    def get_items_by_scene(self, scene_id: int) -> list[Any]:
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

    def clear_items_in_latest_timestep(self, timestep_id: int) -> list[Any]:
        """
        Clear all items in the index with timestep equal to `timestep_id`.

        `timestep_id` is provided to ensure it is valid for this index - this
        function only supports removing items in the latest timestep.

        The scene index will also be updated - in particular, if removing the
        latest timestep means the latest scene is now empty, it is invalid and
        will be removed as well.

        If the given timestep has no items, return an empty list.

        # Args
            timestep_id: The ID of the timestep to query over. Should be the
                latest timestep.
        """
        # Case where this index may not have the requisite number of entries.
        if timestep_id >= len(self._item_timestep_index):
            return []

        # Otherwise it should be the last index only.
        if timestep_id != (len(self._item_timestep_index) - 1):
            raise ValueError(
                "This function only supports removing items for the latest "
                "timestep."
            )

        # Fetch the indices of the cleared items to return.
        subset = self._extract_objects_by_id(
            id=timestep_id, index=self._item_timestep_index
        )
        cleared_items = self._item_buffer[subset]

        # Need to delete stuff in the buffer, not just remove offsets in the
        # timestep and scene indices. This is because adding items appends to
        # the buffer, which means there are assumptions about the buffer's size.
        k = subset.stop - subset.start
        for _ in range(k):
            self._item_buffer.pop()
        self._item_timestep_index.pop()

        if len(self._item_timestep_index) == 0:
            # First case - where the timestep index is now empty.
            self._item_scene_index.pop()
            # This should be empty - otherwise bug in the logic somewhere.
            assert len(self._item_scene_index) == 0
            return cleared_items

        if len(self._item_scene_index) == 1:
            # Second case - where there's only a single scene, so just set the
            # value to the value of the now-latest timestep.
            self._item_scene_index[0] = self._item_timestep_index[-1]
            return cleared_items

        # General case - latest timestep should align with latest scene.

        # If the second-latest scene is equal to the now-latest timestep, then
        # the latest scene is now empty (and invalid) and should be popped.
        if self._item_scene_index[-2] == self._item_timestep_index[-1]:
            self._item_scene_index.pop()
            return cleared_items

        # Otherwise it's not the only timestep for this scene - the scene just
        # needs to have its value set to the value of the now-latest timestep.
        self._item_scene_index[-1] = self._item_timestep_index[-1]

        return cleared_items

    def clear_items_in_latest_scene(self, scene_id: int) -> list[Any]:
        """
        Clear all items in the index with scene equal to `scene_id`.

        `scene_id` is provided to ensure it is valid for this index - this
        function only supports removing items in the latest scene.

        The timestep index will also be updated - in particular, if removing the
        latest scene leads to dummy timestep offsets that span more than the new
        latest scene, those are invalid and will be removed as well.

        If the given scene has no items, return an empty list.

        # Args
            scene_id: The ID of the scene to query over. Should be the latest
                scene.
        """
        # Case where this index may not have the requisite number of entries.
        if scene_id >= len(self._item_scene_index):
            return []

        # Otherwise it should be the last index only.
        if scene_id != (len(self._item_scene_index) - 1):
            raise ValueError(
                "This function only supports removing items for the latest "
                "scene."
            )

        # Fetch the indices of the cleared items to return.
        subset = self._extract_objects_by_id(
            id=scene_id, index=self._item_scene_index
        )
        cleared_items = self._item_buffer[subset]

        # Need to delete stuff in the buffer, not just remove offsets in the
        # timestep and scene indices. This is because adding items appends to
        # the buffer, which means there are assumptions about the buffer's size.
        k = subset.stop - subset.start
        for _ in range(k):
            self._item_buffer.pop()
        self._item_scene_index.pop()

        if len(self._item_scene_index) == 0:
            # First case - where the scene index is now empty. So no timesteps
            # can be present.
            self._item_timestep_index.clear()
            return cleared_items

        # General case - just iterate through the timesteps until one matches
        # the new latest scene.
        while self._item_timestep_index[-1] != self._item_scene_index[-1]:
            self._item_timestep_index.pop()

        # Then remove dummy timesteps - which is checked by seeing if
        # successive timestep offsets have the same value.
        # Account for the case where only one timestep remains.
        second_last_timestep = (
            self._item_timestep_index[-2]
            if len(self._item_timestep_index) > 1
            else self._item_timestep_index[-1] - 1
        )
        while second_last_timestep == self._item_timestep_index[-1]:
            self._item_timestep_index.pop()
            second_last_timestep = (
                self._item_timestep_index[-2]
                if len(self._item_timestep_index) > 1
                else self._item_timestep_index[-1] - 1
            )
        return cleared_items
