import cv2
import numpy as np
from tracking_markers.utils import find_markers, search_window_size_default, marker_template_size_default, upscaling_factor_default, step_size_default
import argparse
from pathlib import Path


def select_markers(video_path: str, frame=0, ROI_X=(0, -1), ROI_Y=(0, -1)):
    """Manually select markers in a video.

    Args:
        video_path (str): Path to the video file.
        frame (int, optional): Frame number to select the markers from. Defaults to 0.
        ROI_X (tuple[int, int], optional): ROI in the x-direction. If -1 is provided, the whole frame will be used. Defaults to (0, -1).
        ROI_Y (tuple[int, int], optional): ROI in the y-direction. If -1 is provided, the whole frame will be used. Defaults to (0, -1).

    Returns:
        np.ndarray: Array of shape (n_markers, 2) containing the marker positions in pixels.
    """

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, frame = cap.read()
    # Flip y-axis in image to match physical frame.
    frame = cv2.flip(frame, 0)
    ROI_X = (ROI_X[0], ROI_X[1] if ROI_X[1] > 0 else frame.shape[1])
    ROI_Y = (ROI_Y[0], ROI_Y[1] if ROI_Y[1] > 0 else frame.shape[0])
    flipped_ROI_Y = (frame.shape[0] - ROI_Y[1], frame.shape[0] - ROI_Y[0])
    ROI_XY = (ROI_X, flipped_ROI_Y)
    frame = frame[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]

    # Collect marker positions from the user by clicking on the image
    markers = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            markers.append((x, y))
            cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

    cv2.namedWindow('Select Markers', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Markers', mouse_callback)

    print("Select markers by clicking on the image. Press 'q' to finish.")
    while True:
        cv2.imshow('Select Markers', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return np.array(markers)


def track_points(
        video_path: str,
        markers: np.ndarray,
        ROI_X=(0, -1),
        ROI_Y=(0, -1),
        frame_range=(0, -1),
        step_size=step_size_default,
        # Parameters for cross-correlation
        search_window_size=search_window_size_default,
        marker_template_size=marker_template_size_default,
        upscaling_factor=upscaling_factor_default,
        template_update_rate=0,
        search_window_update_rate=1,
        # Parameters for visualization
        show_tracked_frame=True,):
    """Track markers in a video.

    Args:
        video_path (str): Path to the video file.
        markers (np.ndarray): Array of shape (n_markers, 2) containing the initial marker positions in pixels.
        ROI_X (tuple[int, int], optional): ROI in the x-direction. If -1 is provided, the whole frame will be used. Defaults to (0, -1).
        ROI_Y (tuple[int, int], optional): ROI in the y-direction. If -1 is provided, the whole frame will be used. Defaults to (0, -1).
        frame_range (tuple, optional): Range of frames to track. Defaults to (0, -1).
        step_size (int, optional): Step size for tracking. Defaults to 1 i.e. each frame is tracked.
        search_window_size (int, optional): Size of the search window. Default is 40px.
        marker_template_size (int, optional): Size of the marker template. Default is 20px.
        upscaling_factor (int, optional): Upscaling factor for the marker template. Defaults to 5.
        template_update_rate (int, optional): Rate at which the template is updated in number of steps. Default is 0 (i.e. no update).
        search_window_update_rate (int, optional): Rate at which the search window is updated in number of steps. Defaults to 1.
        show_tracked_frame (bool, optional): Whether to show the tracked frame. Defaults to True.

    Returns:
        np.ndarray: Array of shape (n_frames, n_markers, 2) containing the marker positions for each frame in pixels.
    """

    cap = cv2.VideoCapture(video_path)
    frame_start, frame_end = frame_range
    frame_number = frame_start
    frame_end = frame_end if frame_end > 0 else int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

    if show_tracked_frame:
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # Initialize the positions of the markers
    template_markers = np.array(markers).astype(np.float64)
    search_markers = template_markers.copy()
    current_markers = template_markers.copy()
    # Initialize the template frame
    _, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    # Flip y-axis in image to match physical frame.
    frame = cv2.flip(frame, 0)
    ROI_X = (ROI_X[0], ROI_X[1] if ROI_X[1] > 0 else frame.shape[1])
    ROI_Y = (ROI_Y[0], ROI_Y[1] if ROI_Y[1] > 0 else frame.shape[0])
    flipped_ROI_Y = (frame.shape[0] - ROI_Y[1], frame.shape[0] - ROI_Y[0])
    ROI_XY = (ROI_X, flipped_ROI_Y)
    frame = frame[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_frame = gray_frame.copy()
    current_frame = gray_frame.copy()

    # Initialize the history of the markers
    markers_history = np.zeros(
        ((frame_end - frame_start) // step_size + 1, len(markers), 2))

    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()

        # Skip frame according to step size
        if (frame_number - frame_start) % step_size != 0:
            frame_number += 1
            continue

        if ret and frame_number <= frame_end:

            # Print frame number
            print(f"Frame #{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")

            # Flip y-axis in image to match physical frame.
            frame = cv2.flip(frame, 0)
            frame = frame[ROI_XY[1][0]: ROI_XY[1]
                          [1], ROI_XY[0][0]: ROI_XY[0][1]]
            # Convert the frame to grayscale
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute current marker positions
            current_markers = find_markers(
                template_frame,
                current_frame,
                template_markers,  # Used for extracting templates around the markers in the template frame
                search_markers,  # Used for placing the search window around the markers in the current frame
                search_window_size=search_window_size,
                marker_template_size=marker_template_size,
                upscaling_factor=upscaling_factor
            )
            # Record the marker positions
            markers_history[(frame_number - frame_start) //
                            step_size] = current_markers

            # Update the template frame
            if template_update_rate != 0 and ((frame_number - frame_start)//step_size) % template_update_rate == 0:
                template_frame = current_frame.copy()
                template_markers = current_markers.copy()

            # Update the search window
            if search_window_update_rate != 0 and ((frame_number - frame_start)//step_size) % search_window_update_rate == 0:
                search_markers = current_markers.copy()

            if show_tracked_frame:
                # Draw the markers on the frame
                for marker_position in current_markers:
                    cv2.drawMarker(frame, marker_position.astype(
                        np.int32), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                # Show the frame and wait for key press
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Update the frame number
            frame_number += 1

        else:
            cap.release()
            break

    cv2.destroyAllWindows()

    return markers_history


def main():
    parser = argparse.ArgumentParser(
        prog="tracking_points.py",
        description="Track markers in a video file using cross-correlation of a template around the markers."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("-r", "--frame_range", type=int, default=(0, -1), nargs=2,
                        help="Range of frames to track. If 0 -1 is provided, the whole video will be used.")
    parser.add_argument("-m", "--markers_path", type=str, default=None,
                        help="Path to the markers file (.npy). If not provided, the user will be prompted to manually select the markers.")
    parser.add_argument("-x", "--ROI_X", type=int, default=(0, -1), nargs=2,
                        help="ROI in the x-direction. If -1 is provided, the whole frame will be used.")
    parser.add_argument("-y", "--ROI_Y", type=int, default=(0, -1), nargs=2,
                        help="ROI in the y-direction. If -1 is provided, the whole frame will be used.")
    parser.add_argument("-ss", "--step_size", type=int, default=1,
                        help="Step size for tracking. Defaults to 1 i.e. each frame is tracked.")
    parser.add_argument("-w", "--search_window_size", type=int, default=search_window_size_default,
                        help="Size of the search window. Default is 40px.")
    parser.add_argument("-t", "--marker_template_size", type=int, default=marker_template_size_default,
                        help="Size of the marker template. Default is 20px.")
    parser.add_argument("-u", "--upscaling_factor", type=int, default=upscaling_factor_default,
                        help="Upscaling factor for the marker template. Defaults to 5.")
    parser.add_argument("-tr", "--template_update_rate", type=int, default=0,
                        help="Rate at which the template is updated in number of steps. Default is 0 (i.e. no update).")
    parser.add_argument("-wr", "--search_window_update_rate", type=int, default=1,
                        help="Rate at which the search window is updated in number of steps. Defaults to 1.")
    parser.add_argument("-ht", "--hide_tracked_frame", action="store_true",
                        default=False, help="Do not show the tracked frame.")
    parser.add_argument("-s", "--save", action="store_true", default=False)
    parser.add_argument("-o", "--out_path", type=str,
                        default="markers_history.npy")
    args = parser.parse_args()

    if args.markers_path is not None:
        # Load the markers from the file
        markers = np.load(args.markers_path)
    else:
        # Manually select the markers
        markers = select_markers(
            args.video_path, frame=args.frame_range[0], ROI_X=args.ROI_X, ROI_Y=args.ROI_Y)

    if len(markers) == 0:
        raise ValueError("No markers selected!")
    # Track the markers
    markers_history = track_points(
        args.video_path,
        markers,
        ROI_X=args.ROI_X,
        ROI_Y=args.ROI_Y,
        frame_range=args.frame_range,
        step_size=args.step_size,
        search_window_size=args.search_window_size,
        marker_template_size=args.marker_template_size,
        upscaling_factor=args.upscaling_factor,
        template_update_rate=args.template_update_rate,
        search_window_update_rate=args.search_window_update_rate,
        show_tracked_frame=not args.hide_tracked_frame
    )

    if args.save:
        print("Saving markers history at", args.out_path)
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, markers_history)


# entrypoint for cli invocation
if __name__ == '__main__':
    main()
