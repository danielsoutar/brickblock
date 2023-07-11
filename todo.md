# TODO

V1

DOD for each of these: unit tests (visualisation involves manually checking in matplotlib)

`~` means in progress

`>` means source written but not tested yet

`X` obviously means fully done (checkbox)

- [X] Create basic `Cube` object
- [X] Correctly visualise a `Cube` in matplotlib
- [X] Create basic `Space` object
- [>] Correctly visualise a single-scene `Space` in matplotlib (single `Cube`)
- [ ] Release as pip package (first use the test pypi index)
- [ ] Add multiple scenes to a `Space` (via `snapshot`)
- [ ] Correctly visualise a multi-scene `Space` in matplotlib (multiple `Cube`s, one per scene)
- [ ] Add multiple `Cube`s in a single scene
- [ ] Correctly visualise a single-scene `Space` in matplotlib (multiple `Cube`s)
- [ ] Correctly visualise a multi-scene `Space` in matplotlib (multiple `Cube`s)
  - Need only visualise individual frames with matplotlib for now.
- [ ] Add support for visual customisations (i.e. updating `cuboid_visual_metadata` and forwarding args)
- [ ] Add support for composite objects (multiple cubes in a single timestep)
- [ ] Add support for `Cuboid`s (any 6-sided polyhedron)
- [ ] Add support for named objects
- [ ] Allow deleting an object (by coordinate, name, timestep ID)
  - Implies that addition/deletion need to contain all data for the object somewhere, or have a column to mask it. Timestep ID should be fine though.
- [ ] Allow deleting multiple objects (by scene ID)
  - As above
- [ ] Allow mutating an object's visual metadata (by coordinate, name, timestep ID, scene ID)
- [ ] Allow creating a object by offset from a given object (by coordinate, name, timestep ID)
  - If allowing by scene ID, that implies multiple objects in a single timestep. Unless you offset each object in terms of timestep as well.
- [ ] Allow mutating an object by affine transform (by name, timestep ID, scene ID)
  - Need to figure out whether these 'happen-before' or 'happen-after', whether to include inverse transform, etc.
- [ ] Allow mutating an object by affine transform (by coordinate ID)
  - This could be trickier and potentially not needed.
- [ ] Create GIF from a multi-scene `Space`
- [ ] Log debug info per scene (objects added/mutated/deleted, possibly camera orientation as well)

V2

- [ ] Use LLM for natural language instructions instead of coding things (how the fuck that would work I have no idea)
- [ ] Allow interactive modification of a rendered scene (how the fuck that would work I still have no idea)
- [ ] Allow other kinds of objects (text with cool-looking connecting lines, triangles, spheres)
- [ ] Allow interactive playback of the GIF created by a `Space` (i.e. be able to pause/play/go back a slide, inside a notebook)
- [ ] Use a different backend other than matplotlib?
- [ ] RAY TRACING ALL THE THINGS (why!?)