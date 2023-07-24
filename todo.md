# TODO

V1

DOD for each of these: unit tests (visualisation involves manually checking in matplotlib)

20/27 done

- [X] Create basic `Cube` object
- [X] Correctly visualise a `Cube` in matplotlib
- [X] Create basic `Space` object
- [X] Correctly visualise a single-scene `Space` in matplotlib (single `Cube`)
- [X] Release as pip package (first use the test pypi index)
- [X] Add multiple scenes to a `Space` (via `snapshot`)
- [X] Correctly visualise a multi-scene `Space` in matplotlib (multiple `Cube`s, one per scene)
- [X] Add multiple `Cube`s in a single scene
- [X] Correctly visualise a single-scene `Space` in matplotlib (multiple `Cube`s)
- [X] Correctly visualise a multi-scene `Space` in matplotlib (multiple `Cube`s)
  - Need only visualise individual frames with matplotlib for now.
- [X] Have `render` imply the current timestep as a `Scene`, if not already marked as such.
- [X] Add support for visual customisations (i.e. updating `cuboid_visual_metadata` and forwarding args)
- [X] Add support for composite objects (multiple cubes in a single timestep)
- [X] Fix bug where composite objects are swapping the height and depth dimensions
  - This is a more general problem - it also occurs for arbitrary objects.
- [X] Add support for `Cuboid`s (any 6-sided polyhedron)
- [X] Add support for named objects
- [X] Allow mutating an object's visual metadata by coordinate
- [X] Allow mutating an object's visual metadata by name
- [X] Allow mutating an object's visual metadata by timestep ID
- [X] Allow mutating an object's visual metadata by scene ID
- [ ] Work out a way of positioning the camera better
- [ ] Allow creating a object by offset from a given object (by coordinate, name, timestep ID)
  - If allowing by scene ID, that implies multiple objects in a single timestep. Unless you offset each object in terms of timestep as well.
- [ ] Allow mutating an object by affine transform (by name, timestep ID, scene ID)
  - Need to figure out whether these 'happen-before' or 'happen-after', whether to include inverse transform, etc.
- [ ] Allow mutating an object by affine transform (by coordinate ID)
  - This could be trickier and potentially not needed.
- [ ] Distinguish between 'scene' and 'non-scene' timesteps
- [ ] Create GIF from a multi-scene `Space`
- [ ] Log debug info per scene (objects added/mutated/deleted, possibly camera orientation as well)

V2

- [ ] Use LLM for natural language instructions instead of coding things (how the fuck that would work I have no idea)
- [ ] Allow interactive modification of a rendered scene (how the fuck that would work I still have no idea)
- [ ] Allow other kinds of objects (text with cool-looking connecting lines, triangles, spheres)
- [ ] Allow interactive playback of the GIF created by a `Space` (i.e. be able to pause/play/go back a slide, inside a notebook)
- [ ] Use a different backend other than matplotlib?
- [ ] RAY TRACING ALL THE THINGS (why!?)