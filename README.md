# Sterling: A fun visualisation library for those that like boxes

This is a small library that uses blocks in matplotlib's visually appealing 3D extension - and aims to be the 'seaborn of matplotlib-3D', if you will.

## Core abstractions

At the centre of Sterling is the `Space`. A `Space` contains objects, and when a user wants a visualisation, they render the current state of the `Space` - the rendered state is known as a `Scene`.

Currently there is one object used for composing visualisations in Sterling: the `Cube`. `Cube` objects can be added into the `Space` with a degree of control over their visual presentation: transparency, colour, line widths.

Sterling also supports `Transition`s and `Transform`s:

* A `Transform` is simply any change to the space like adding one or more `Cube` objects. It does not produce a `Scene`. These are useful when you are just iteratively building up to a complex `Scene`.
* A `Transition`, by contrast, is a transform between two `Scene`s. These are useful for generating sequences of `Scene`s - the `Space` will keep track of its state over time, as well as which states are `Scene`s. A `Transition` can be defined by one or more `Transform`s.

Having these abstractions allows programmers to create animated visualisations like GIFs. You define a `Space`, using `Transform`s and `Transition`s to evolve the state, and the `Scene` objects are persisted to enable sequences of images for use in GIFs.

```python
import brickblock as bb


# Let's start with adding a few objects to our space!
cuboid_base = np.array((0, 0, 0))
cuboid_dims = np.array((4, 3, 4))
cuboid_opts = {'color': None, 'alpha': 0}
input_cube = bb.Cuboid(base, cuboid_dims, **cuboid_opts)

filter_base = np.array((0, 0, 1))
filter_dims = np.array((3, 3, 3))
filter_opts = {'color': (1., 1., 0.5), 'alpha': 0.3, 'linewidths': 0.3}
filter_cube = bb.Cube(filter_base, filter_dims, **filter_opts)

s = bb.Space()
# Add a 4x3x4 rectangular cuboid to the space.
s = bb.add_transform(s, input_cube)
# Add a 3x3x3 filter to the space with a single cube.
s = bb.add_transform(s, filter_cube)

# Render the state as a scene and use in a plot-like way...
fig, ax = bb.render(s)
# Or take a snapshot to indicate that this is a scene...
state_history = bb.snapshot(s)
# Do more stuff...
# And then when you're ready, generate a GIF-like output...
gif_filename = bb.stream(s, "gif", path_to_file))
```
