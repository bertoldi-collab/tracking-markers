# A humble image tracking code

This is a humble image tracking code.
It is humble because it does what it can.

## Installation

Assuming you have access to the repo and ssh keys are set up in your GitHub account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/tracking-markers.git@main
```

## Some info

- It is based on the [OpenCV](https://opencv.org/) library.
- Allows for markers to be manually selected or an `np.ndarray` of markers can be loaded from a file.
- Works best on high-contrast videos.
