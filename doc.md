# Important stuff while I make actual docs

## Examples

The following is a basic example in Brickblock of creating a simple scene:

```python
def main() -> None:
    space = bb.Space()
    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 0, 0]),
            w=3,
            h=3,
            d=3,
            name="input-tensor"
        )
    )

    space.add_composite(
        bb.CompositeCube(
            base_vector=np.array([0, 1, 0]),
            w=2,
            h=2,
            d=3,
            name="filter-tensor",
            facecolor="yellow",
            alpha=0.3,
        )
    )

    fig, ax = space.render()
    plt.show()

```

## Scenes and Transforms

TODO: Fill in

### The Camera

In a `Scene`, the camera is re-positioned from its default in matplotlib. This is because the camera in matplotlib by default centers around the zero point regardless of what gets added to the axes. Brickblock addresses this limitation by implicitly inserting an invisible bounding box around all the objects in the space to the axes object, and updating the view accordingly. The behaviour currently is not ideal or going to work well in all scenarios, but this can be improved in the future.

It is important to note however that this bounding box is around **all** objects in the space - including invisible ones! In the future this behaviour may be made configurable, or objects could be excluded from this behaviour.

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