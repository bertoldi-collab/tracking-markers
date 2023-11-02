# A humble image tracking code

This is a humble image tracking code.
It is humble because it does what it can.

## Installation

Assuming you have access to the repo and ssh keys are set up in your GitHub account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/tracking-markers.git@main
```

## How to use

### CLI

Just run

```bash
tracking-markers path/to/video.mp4
```

See `tracking-markers --help` for more info on all the options.

### Python

The main module is [`tracking_points.py`](tracking_markers/tracking_points.py) defining the `track_points(...)` function that actually does the tracking of given video and the function `select_markers(...)` that allows for manual selection of markers.
These functions can be used independently.
The file [`tracking_points.py`](tracking_markers/tracking_points.py) can also be used as a script.

## Some info

- It is based on the [OpenCV](https://opencv.org/) library.
- Allows for markers to be manually selected or an `np.ndarray` of markers can be loaded from a file.
- Works best on high-contrast videos.
