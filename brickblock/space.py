from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from brickblock.cube import Cube, CompositeCube


class SpaceStateChange:
    ...


@dataclass
class Addition(SpaceStateChange):
    timestep_id: int
    name: str | None


@dataclass
class Mutation(SpaceStateChange):
    name: str | None
    primitive_id: int | None
    timestep_id: int | None
    scene_id: int | None
    subject: np.ndarray | tuple[dict[str, Any], dict[str, Any]]


@dataclass
class Deletion(SpaceStateChange):
    timestep_id: int
    name: str | None


class Space:
    """
    Representation of a 3D cartesian coordinate space, which tracks its state
    over time.

    This class contains geometric objects for plotting, and acts as a wrapper
    over the visualisation library.

    cuboid_coordinates contains the raw spatial data of each cuboid in the
    space. As coordinates are in 3D space and all objects are 6-sided, that
    means the data structure is a (Nx6x4x3) np array. It is dynamically resized
    as needed.

    cuboid_visual_metadata contains the visual metadata about each cuboid in the
    space. Since it contains heterogenous data, a DataFrame makes more sense
    than a numpy array. Essentially, everything of the form
    `polycollection.setXXX(...)` is stored here.

    cuboid_index contains the indexing for each cuboid in the space. For
    instance, all cuboids added have their own `primitive_counter` ID. They also
    have a `timestep` ID which is unique to each 'full' object added. For
    instance, a composite object consisting of many cuboids might be added, and
    each cuboid within would have the same `timestep` ID. Finally, all cuboids
    also have a Scene ID, which corresponds to the current scene.

    changelog represents the change to state in each transform to the space.
    There are three main categories of changes:
        - addition
        - mutation
        - deletion
    For addition:
        - an Addition object represents what was added, using the timestep ID
        (and optionally a name).
        - Its converse is a Deletion with the same data.
    For mutation:
        - a Mutation object represents what was changed, using an identifier
        (either a name, or a timestep, or a scene ID) and a
        subject (either a translation or visual change) with a before and after.
        - Its converse is a Mutation with the same ID and the subject
        with before and after swapped (affine transform still to be inverted).
    For deletion:
        - Complement of addition, see above.
    """

    dims: np.ndarray
    mean: np.ndarray
    total: np.ndarray
    num_objs: int
    primitive_counter: int
    time_step: int
    scene_counter: int
    cuboid_coordinates: np.ndarray
    cuboid_visual_metadata: dict[str, list]
    cuboid_index: dict[int, dict[int, list[int]]]
    changelog: list[SpaceStateChange]

    def __init__(self) -> None:
        self.dims = np.zeros((3, 2))
        self.mean = np.zeros((3, 1))
        self.total = np.zeros((3, 1))
        self.num_objs = 0
        self.primitive_counter = 0
        self.time_step = 0
        self.scene_counter = 0
        self.cuboid_coordinates = np.zeros((10, 6, 4, 3))
        self.cuboid_visual_metadata = {}
        self.cuboid_index = {0: {}}
        self.changelog = []

    def add_cube(self, cube: Cube) -> None:
        """
        TODO: Fill in.
        """
        cube_bounding_box = cube.get_bounding_box()
        cube_mean = np.mean(cube.points(), axis=0).reshape((3, 1))

        self.total += cube_mean
        self.num_objs += 1
        self.mean = self.total / self.num_objs

        if self.num_objs == 1:
            dim = cube_bounding_box
        else:
            # Since there are multiple objects, ensure the resulting dimensions
            # of the surrounding box are centred around the mean.
            dim = np.array(
                [
                    [
                        min(self.dims[i][0], cube_bounding_box[i][0]),
                        max(self.dims[i][1], cube_bounding_box[i][1]),
                    ]
                    for i in range(len(cube_bounding_box))
                ]
            ).reshape((3, 2))

        self.dims = dim

        current_no_of_entries = self.cuboid_coordinates.shape[0]
        if self.primitive_counter >= current_no_of_entries:
            # refcheck set to False since this avoids issues with the debugger
            # referencing the array!
            self.cuboid_coordinates.resize(
                (2 * current_no_of_entries, *self.cuboid_coordinates.shape[1:]),
                refcheck=False,
            )

        self.cuboid_coordinates[self.primitive_counter] = cube.faces
        for key, value in cube.get_visual_metadata().items():
            if key in self.cuboid_visual_metadata.keys():
                self.cuboid_visual_metadata[key].append(value)
            else:
                self.cuboid_visual_metadata[key] = [value]

        def add_key_to_nested_dict(d, keys):
            for key in keys:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d.setdefault(keys[-1], -1)

        keys = [self.scene_counter, self.time_step]
        add_key_to_nested_dict(self.cuboid_index, keys)
        self.cuboid_index[self.scene_counter][self.time_step] = [
            self.primitive_counter
        ]

        self.changelog.append(Addition(self.time_step, None))

        self.primitive_counter += 1
        self.time_step += 1

    def add_composite(self, composite: CompositeCube) -> None:
        cube_bounding_box = cube.get_bounding_box()
        cube_mean = np.mean(cube.points(), axis=0).reshape((3, 1))

        self.total += cube_mean
        self.num_objs += 1
        self.mean = self.total / self.num_objs

        if self.num_objs == 1:
            dim = cube_bounding_box
        else:
            # Since there are multiple objects, ensure the resulting dimensions
            # of the surrounding box are centred around the mean.
            dim = np.array(
                [
                    [
                        min(self.dims[i][0], cube_bounding_box[i][0]),
                        max(self.dims[i][1], cube_bounding_box[i][1]),
                    ]
                    for i in range(len(cube_bounding_box))
                ]
            ).reshape((3, 2))

        self.dims = dim

        current_no_of_entries = self.cuboid_coordinates.shape[0]
        if self.primitive_counter >= current_no_of_entries:
            # refcheck set to False since this avoids issues with the debugger
            # referencing the array!
            self.cuboid_coordinates.resize(
                (2 * current_no_of_entries, *self.cuboid_coordinates.shape[1:]),
                refcheck=False,
            )

        self.cuboid_coordinates[self.primitive_counter] = cube.faces
        for key, value in cube.get_visual_metadata().items():
            if key in self.cuboid_visual_metadata.keys():
                self.cuboid_visual_metadata[key].append(value)
            else:
                self.cuboid_visual_metadata[key] = [value]

        def add_key_to_nested_dict(d, keys):
            for key in keys:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d.setdefault(keys[-1], -1)

        keys = [self.scene_counter, self.time_step]
        add_key_to_nested_dict(self.cuboid_index, keys)
        self.cuboid_index[self.scene_counter][self.time_step] = [
            self.primitive_counter
        ]

        self.changelog.append(Addition(self.time_step, None))

        self.primitive_counter += 1
        self.time_step += 1

    def snapshot(self) -> None:
        if self.scene_counter not in self.cuboid_index.keys():
            raise Exception(
                "A snapshot must include at least one addition, mutation, or "
                "deletion in the given scene."
            )
        self.scene_counter += 1

    # TODO: Decide whether passing the Axes or having it be fully constructed by
    # brickblock is a good idea.
    # TODO: It seems controlling the azimuth and elevation parameters (which are
    # handily configurable!) is what you need for adjusting the camera.
    # TODO: Calling plt.show shows each figure generated by render(), rather than
    # only the last one (though it shows the last one first). Can this be fixed?
    def render(self) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None
        )
        ax = fig.add_subplot(111, projection="3d")
        # Remove everything except the objects to display.
        ax.set_axis_off()

        # This was done *after* adding the polycollections in the original
        # notebook, but you've always conceptualised it as being before.
        # Maybe you did do that, but it means less flexibility in the plots? Who
        # knows...
        # bounds = self.dims
        # bound_max = np.max(bounds)

        # Does this always work? Does this ever work?
        # ax.set_xlim(-bound_max / 8, bound_max * 1)
        # ax.set_ylim(-bound_max / 4, bound_max * 0.75)
        # ax.set_zlim(-bound_max / 4, bound_max * 0.75)

        for timestep in range(self.time_step):
            # Create the object for matplotlib ingestion.
            matplotlib_like_cube = Poly3DCollection(
                self.cuboid_coordinates[timestep]
            )
            # Set the visual properties first - check if these can be moved into
            # the Poly3DCollection constructor instead.
            visual_properties = {
                k: self.cuboid_visual_metadata[k][timestep]
                for k in self.cuboid_visual_metadata.keys()
            }
            matplotlib_like_cube.set_facecolor(visual_properties["facecolor"])
            matplotlib_like_cube.set_linewidths(visual_properties["linewidth"])
            matplotlib_like_cube.set_edgecolor(visual_properties["edgecolor"])
            matplotlib_like_cube.set_alpha(visual_properties["alpha"])
            ax.add_collection3d(matplotlib_like_cube)

        return fig, ax

        # Ideally you'd have a numpy array for all the cubes (cube_data),
        # a pandas dataframe for the polycollection metadata (cube_metadata),
        # with an index that ties the two (a 'timestep' that is broadcast for
        # grouped objects, and an incrementing ID that uniquely identifies each
        # primitive).
        # This is fine for adding objects, but what happens when a user wants to
        # hide/modify/delete an object, for instance?
        # There are a few choices:
        #
        # a) Identify a cube/cuboid by its coordinates (allows duplicates)
        # b) Identify a cube/cuboid by a name (unique)
        # c) Identify a cube/cuboid by timestep (allows duplicates)
        # d) Identify a cube/cuboid by scene (allows duplicates)
        #
        # (a) is easy enough for a cube, you can search cube_data by 'row'. But
        # a composite would be harder - maybe the index or metadata could store
        # overall shape per object inserted. Or even its own thing potentially.
        # (b) That would be useful - again, either add a name field in the index
        # or metadata.
        # (c) That should be straightforward - you can just query the index.
        # (d) is easy enough - just range over the scene ID.
        #
        # All of these can be useful. But you can't update in-place naively - in
        # order to preserve history. So you either need a separate data
        # structure for tracking changes, or you need to add a new object to the
        # data structures, possibly marking with a 'scene_id'.

        # This has all the info needed - the very first point can be taken as
        # the base vector. The first and last faces contain all unique points
        # (and conveniently, in order too!).
        # cube.faces,
        # primitive ID - unique for every primitive passed in
        # self.primitive_counter
        # timestep ID - for every 'transaction'. For instance, a composite
        # cuboid would have the same timestep ID for each individual cube within
        # it.
        # self.time_step

        # When updating/modifying/deleting by coordinates (allows duplicates)
        # A vector that lands on or within any cube/cuboid. Could simplify for
        # now and say must be base vector.
        # base_point
        # When updating/modifying/deleting by name (unique)
        # A name - implies support for NamedCube/NamedCuboid (constructor should
        # just be a Cube/Cuboid and a name)
        # name_of_cube
        # When updating/modifying/deleting by timestep (allows duplicates)
        # A timestep. Easy enough.
        # time_step
        # When updating/modifying/deleting by scene (allows duplicates)
        # scene_id

        # If I have the dataframes for coordinate data, metadata, and index,
        # how do I keep the old state and the new state?
        #
        # > Add new entries, with a reference to the previous entry (mem cost,
        #   track probably nullable column)
        # > Actually delete the entry (time cost, need to add largely redundant
        #   state column)
        #
        # I think it makes sense to only add entries, with a 'changelog'
        # representing changes to the internal data. The advantage of the
        # changelog is that you can group various transforms together and
        # batch-execute them between scenes.


# fig = plt.figure(figsize=(12, 10))
# fig.subplots_adjust(
#   left=0, bottom=0, right=1, top=1, wspace=None, hspace=None
# )
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# s = Space()
# s.add_cube(test_cube)
# s.snapshot()
# fig, ax = s.render()
# ax.set_axis_off()
# plt.show()
