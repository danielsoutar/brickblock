# Important stuff while I make actual docs

## Examples

TODO: Fill in

## Scenes and Transforms

TODO: Fill in

## Axes vs Dimensions

NB: By 'Axes', we mean the x-, y-, and z-axis. For dimensions, we mean Height, Width, and Depth.

In 2D Matplotlib, the issue of dimensions versus axes is a non-issue in virtually all cases for sane people. The x-axis corresponds to the width of a plot, the y-axis the height of a plot, and transforming data to swap between these is just a 2D transpose if needed.

However, for 3D, things are less straightforward. The mapping between {X, Y, Z} and {Width, Height, Depth} is ultimately arbitrary, but most (and certainly this author) would probably say the x-axis corresponds to width, the y-axis the height, and the z-axis the depth. Unfortunately, in Matplotlib, the Z axis is treated as Height, with the Y axis treated as Depth, and this is reflected in the GUI that displays plots. This forces users to either:

* Change the camera orientation and position

This is not always possible, and means matplotlib's camera orientation is confusing to users when moving it interactively since it only has two degrees of rotation.

* Change the data

This is usually dissatisfying for users, and is tedious and error-prone.

This is not ideal. So Brickblock addresses this by providing users with the following mapping in its objects:

* x-axis <--> Width
* y-axis <--> Height
* z-axis <--> Depth

Internally, Brickblock will represent data accordingly for matplotlib. But users won't have to care about this (hopefully)! Ideally the library would have its own mapping internally, and the conversion for matplotlib becomes an implementation detail only for code that *specifically* deals with that backend via a transpose or similar. This would better decouple from matplotlib and also allow greater flexibility with other backends that might represent data differently.