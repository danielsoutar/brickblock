# Brickblock: A fun visualisation library for those that like boxes

This is a small library that uses blocks in matplotlib's visually appealing 3D extension - and aims to be the 'seaborn of matplotlib-3D'.

## Core abstractions

At the centre of Brickblock is the `Space`. A `Space` represents a 3D cartesian coordinate space. It contains objects, and when a user wants a visualisation, they render the current state of the `Space` - the rendered state is known as a `Scene`.

There are objects used for composing visualisations in Brickblock, such as the `Cube`. Objects can be added into the `Space` with a degree of control over their visual presentation: transparency, colour, line widths. They can also be mutated - and can be selected by name, base vector, or various IDs.

Having these abstractions allows programmers to more easily create animated 3D visualisations, like GIFs. You define a `Space`, adding and mutating objects to evolve the state, and the `Scene` objects are persisted to enable sequences of images for use in GIFs.

## Contributing

Contributions are more than welcome! There is a very rough TODO in [todo.md](todo.md) that outlines both short- and long-term goals for the library. However, there are some rules:

* Always follow the [code of conduct](CODE_OF_CONDUCT.md)
* All the tests should be passing with your change
* Explain your change (PR template coming soon)
* Add relevant tests and docstrings
* Format your changes with `black`
