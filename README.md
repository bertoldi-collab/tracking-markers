# A humble image tracking code

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=ecf0f1&labelColor=34495e)
[![GitHub license](https://img.shields.io/github/license/bertoldi-collab/tracking-markers?labelColor=34495e)](https://github.com/bertoldi-collab/tracking-markers/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbertoldi-collab%2Ftracking-markers&count_bg=%2327AE60&title_bg=%2334495E&icon=github.svg&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)

This is a humble image tracking code.
It is humble because it does what it can.

<p align="center">
  <img width="460" height="300" src="examples/spaceman.gif">
</p>

## Installation

Assuming you have access to the repo and ssh keys are set up in your GitHub account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/tracking-markers.git@main
```

## How to use

### CLI

Run in a terminal

```bash
tracking-markers path/to/video.mp4
```

See `tracking-markers --help` for more info on all the options.

### Python

The main module is [`tracking_points.py`](tracking_markers/tracking_points.py) defining the `track_points(...)` function that actually does the tracking of a given video and the function `select_markers(...)` that allows the manual selection of markers.
These functions can be used independently.
The file [`tracking_points.py`](tracking_markers/tracking_points.py) can also be used as a script.

## Some info

- It is based on the [OpenCV](https://opencv.org/) library.
- Allows for markers to be manually selected or an `np.ndarray` of markers can be loaded from a file.
- Works best on high-contrast videos.
