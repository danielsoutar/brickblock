from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
# import imageio
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ColourPoint = tuple[float, float, float]


@dataclass
class Cube:
    """
    Primitive object for composing scenes.

    This object is intended for purely as a 'front-end' for users to interact
    with for composing `Scene`s.
    """
    faces: np.ndarray
    facecolor: ColourPoint | None = None
    linewidths: float = 0.1
    edgecolor: str = 'black'
    alpha: float = 0.0

    def __init__(
            self,
            points: np.ndarray,
            facecolor: ColourPoint | None = None,
            linewidths: float = 0.1,
            edgecolor: str = 'black',
            alpha: float = 0.0
    ) -> None:
        # Check dimensions are valid - either 4 points, defined as the three
        # basis vectors from the base point (a 'cube'), or 8 points fully
        # defining the cube.
        using_shorthand = len(points) == 4
        if not using_shorthand and len(points) != 8:
            raise ValueError(
                "Cube objects require either 4 points (a base and three basis "
                "vectors) or 8 points defining the vertices."
            )
        # If using 'shorthands' (i.e. implicitly defining by 3 vectors), expand
        # and construct the full cuboid.
        full_points = self._construct_points(points, using_shorthand)

        self.faces = self._construct_faces(full_points)
        self.face_colour = facecolor
        self.line_width = linewidths
        self.edge_colour = edgecolor
        self.alpha = alpha

    def points(self) -> np.ndarray:
        return np.array([self.faces[0], self.faces[-1]]).reshape((8, 3))

    def get_visual_metadata(self) -> dict[str, Any]:
        return {
            'facecolor': self.facecolor,
            'linewidths': self.linewidths,
            'edgecolor': self.edgecolor,
            'alpha': self.alpha,
        }

    def _construct_points(self, points: np.ndarray, using_shorthand: bool) -> np.ndarray:
        """
        Construct the full set of points from a possibly partial set of points.
        """
        if using_shorthand:
            # Shorthand convention is to have the 'bottom-left-front' point as
            # the base, with points defining height/width/depth of the cube
            # after (using the left-hand rule).
            # NB: in the 'xyz' axes, we have width-height-depth (WHD) for the coordinates.
            base, h, w, d = points
            # Note: the ordering of points matters.
            full_points = np.array(
                [
                    # bottom-left-front
                    base,
                    # bottom-left-back
                    base + d,
                    # bottom-right-back
                    base + w + d,
                    # bottom-right-front
                    base + w,
                    # top-left-front
                    base + h,
                    # top-left-back
                    base + h + d,
                    # top-left-back
                    base + h + w + d,
                    # top-right-front
                    base + h + w,
                ]
            )
        else:
            full_points = points

        return full_points.reshape((8, 3))

    def _construct_faces(self, points: np.ndarray) -> np.ndarray:
        return np.array([
            (points[0], points[1], points[2], points[3]),  # bottom
            (points[0], points[4], points[7], points[3]),  # front face
            (points[0], points[1], points[5], points[4]),  # left face
            (points[3], points[7], points[6], points[2]),  # right face
            (points[1], points[5], points[6], points[2]),  # back face
            (points[4], points[5], points[6], points[7]),  # top
        ]).reshape((6, 4, 3))


# test_points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]).reshape((4, 3))
# test_cube = Cube(test_points)
# poly = Poly3DCollection(test_cube.faces)


def test_plot():
    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return fig, ax


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
    dims: np.ndarray | None
    mean: np.ndarray
    total: np.ndarray
    num_objs: int
    # All the FUN stuff.
    cuboid_coordinates: np.ndarray
    cuboid_visual_metadata: dict[str, list]
    cuboid_index: dict[str, dict[str, dict]]
    changelog: list[SpaceStateChange]


    def __init__(self, dims: np.ndarray | None = None) -> None:
        self.dims = dims
        self.mean = np.zeros((3, 1))
        self.total = np.zeros((3, 1))
        self.num_objs = 0
        self.primitive_counter = 0
        self.time_step = 0
        self.scene_counter = 0
        self.cuboid_coordinates = np.zeros((10, 6, 4, 3))
        self.cuboid_visual_metadata = {}
        self.cuboid_index = defaultdict(lambda: defaultdict(None))
        self.changelog = []


    def add_cube(self, cube: Cube) -> None:
        """
        Make sure the space is large enough to encompass all objects in it.
        This is achieved by ensuring the space is centred around the
        geometric mean of the objects within it.
        """
        cube_bounding_box = get_bounding_box(cube)
        cube_mean = np.mean(cube.points(), axis=0).reshape((3, 1))

        self.total += cube_mean
        self.num_objs += 1

        if self.dims is None:
            dim = cube_bounding_box
        else:
            # Since there are multiple objects, ensure the resulting dimensions
            # of the surrounding box are centred around the mean.
            dim = np.array([
                [
                    min(self.dims[i][0], cube_bounding_box[i][0]),
                    max(self.dims[i][1], cube_bounding_box[i][1])
                ] for i in range(len(cube_bounding_box))
            ]).reshape((3, 2))

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
        for (key, value) in cube.get_visual_metadata().items():
            if key in self.cuboid_visual_metadata.keys():
                self.cuboid_visual_metadata[key].append(value)
            else:
                self.cuboid_visual_metadata[key] = [value]

        self.cuboid_index[self.scene_counter][self.time_step] = [
            self.primitive_counter
        ]

        self.changelog.append(Addition(self.time_step, None))

        self.primitive_counter += 1
        self.time_step += 1

    def snapshot(self) -> None:
        self.scene_counter += 1

    # TODO: Decide whether passing the Axes or having it be fully constructed by
    # brickblock is a good idea.
    # TODO: It seems controlling the azimuth and elevation parameters (which are
    # handily configurable!) is what you need for adjusting the camera.
    def render(self) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

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
            matplotlib_like_cube.set_facecolor(visual_properties['facecolor'])
            matplotlib_like_cube.set_linewidths(visual_properties['linewidths'])
            matplotlib_like_cube.set_edgecolor(visual_properties['edgecolor'])
            matplotlib_like_cube.set_alpha(visual_properties['alpha'])
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
        # > Add new entries, with a reference to the previous entry (mem cost, track probably nullable column)
        # > Actually delete the entry (time cost, need to add largely redundant state column)
        # 
        # I think it makes sense to only add entries, with a 'changelog'
        # representing changes to the internal data. The advantage of the
        # changelog is that you can group various transforms together and
        # batch-execute them between scenes.


def get_bounding_box(cube: Cube) -> np.ndarray:
    """
    Get the bounding box around the given cuboid's `points`.

    The input should be (M x 3), where M is the number of points. Each point
    should be in WHD order (xs, ys, zs).

    The output should be (3, 2), with each row (in WHD order) corresponding to
    the minimum and maximum of the given dimension. 
    """
    points = cube.points()
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    max_range = np.array(
        [x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0

    mid_x = (x_max+x_min) * 0.5
    mid_y = (y_max+y_min) * 0.5
    mid_z = (z_max+z_min) * 0.5

    return np.array([
        [mid_x - max_range, mid_x + max_range],
        [mid_y - max_range, mid_y + max_range],
        [mid_z - max_range, mid_z + max_range]
    ]).reshape((3, 2))


# fig = plt.figure(figsize=(12, 10))
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
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


# # Let's start with adding a few objects to our space!
# cuboid_base = np.array((0, 0, 0))
# cuboid_dims = np.array((4, 3, 4))
# cuboid_opts = {'color': None, 'alpha': 0}

# filter_base = np.array((0, 0, 1))
# filter_dims = np.array((3, 3, 3))
# filter_opts = {'color': (1., 1., 0.5), 'alpha': 0.3, 'linewidths': 0.3}

# s = Space()
# # Add a 4x3x4 rectangular cuboid to the space.
# s = add_transform(s, Cuboid(cuboid_base, cuboid_dims, **cuboid_opts))
# # Add a 3x3x3 filter to the space with a single cube.
# s = add_transform(s, Cube(filter_base, filter_dims, **filter_opts))

# # Render the state as a scene and use in a plot-like way...
# fig, ax = render(s)
# # Or take a snapshot to indicate that this is a scene...
# state_history = snapshot(s)
# # Do more stuff...
# # And then when you're ready, generate a GIF-like output...
# # tempfile.TemporaryFile could be used for the intermediate images
# gif_filename = stream(s, GIF(path_to_file))