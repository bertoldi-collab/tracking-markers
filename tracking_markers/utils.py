import cv2
import numpy as np


# Set default parameter values
step_size_default = 1  # Step size for tracking. Defaults to 1 i.e. each frame is tracked.
# Cross-correlation parameters
search_window_size_default = 40  # Size of the search window. Default is 40px.
marker_template_size_default = 20  # Size of the marker template. Default is 20px.
upscaling_factor_default = 5  # Upscaling factor for the marker template and search window. Default is 5.


def find_markers(
        template_frame: np.ndarray,
        search_frame: np.ndarray,
        template_markers: np.ndarray,
        search_markers: np.ndarray,
        search_window_size=search_window_size_default,
        marker_template_size=marker_template_size_default,
        upscaling_factor=upscaling_factor_default,):
    """Find the markers by cross-correlating the search frame with the template frame.

    Args:
        template_frame (np.ndarray): The frame (grayscale) used to extract templates.
        search_frame (np.ndarray): The frame (grayscale) used for search.
        template_markers (np.ndarray): The positions of the markers in the template frame in pixel coordinates.
        search_markers (np.ndarray): The positions of the markers in the search frame in pixel coordinates.
        search_window_size (int): The size of the search window. Defaults to 40px.
        marker_template_size (int): The size of the marker template. Defaults to 20px.
        upscaling_factor (int): The upscaling factor for the marker template and search window (used to reduce noise in the cross-correlation). Defaults to 5.

    Returns:
        np.ndarray: Array of shape (n_markers, 2) of the new positions of the markers in the search frame.
    """

    current_markers = search_markers.copy()

    # Loop over the previous marker positions and find the corresponding marker in the current frame
    for (i, template_marker), search_marker in zip(enumerate(template_markers), search_markers):
        # Marker position in the previous frame
        x, y = template_marker
        x_search, y_search = search_marker

        # Define marker template centered on the previous position
        marker_template = template_frame[
            int(max(y - marker_template_size/2, 0)):int(min(y + marker_template_size/2, template_frame.shape[0])),
            int(max(x - marker_template_size/2, 0)):int(min(x + marker_template_size/2, template_frame.shape[1]))
        ]
        # Define the search window centered on the previous marker position
        search_window = search_frame[
            int(max(y_search - search_window_size/2, 0)):int(min(y_search + search_window_size/2, search_frame.shape[0])),
            int(max(x_search - search_window_size/2, 0)):int(min(x_search + search_window_size/2, search_frame.shape[1]))
        ]

        # Upscale the marker template and search window to reduce noise in the template matching
        try:
            marker_template = cv2.resize(
                marker_template,
                (int(marker_template.shape[0]*upscaling_factor), int(marker_template.shape[1]*upscaling_factor)),
                interpolation=cv2.INTER_CUBIC
            )
        except:
            raise Exception(
                f"Marker template is {marker_template.shape[0]}x{marker_template.shape[1]}px. Marker at position {template_marker} could not be found.")
        try:
            search_window = cv2.resize(
                search_window,
                (int(search_window.shape[0]*upscaling_factor), int(search_window.shape[1]*upscaling_factor)),
                interpolation=cv2.INTER_CUBIC
            )
        except:
            raise Exception(
                f"Search window is {search_window.shape[0]}x{search_window.shape[1]}px. Marker at position {template_marker} could not be found.")
        # Compute the cross-correlation between the marker template and the search window
        # Catch exception if the template is larger than the search window
        try:
            xcorr_result = cv2.matchTemplate(search_window, marker_template, cv2.TM_CCORR_NORMED)
        except:
            raise Exception(
                f"Marker template is {marker_template.shape[0]}x{marker_template.shape[1]}px. Search window is {search_window.shape[0]}x{search_window.shape[1]}px. Marker at position {template_marker} could not be found."
            )

        # Get the position of the marker in the current frame
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(xcorr_result)
        current_markers[i] = np.array([
            x_search + (marker_template.shape[0]/2 - search_window.shape[0]/2 + max_loc[0])/upscaling_factor,
            y_search + (marker_template.shape[1]/2 - search_window.shape[1]/2 + max_loc[1])/upscaling_factor
        ])

    return current_markers
