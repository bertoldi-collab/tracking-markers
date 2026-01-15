import cv2
import numpy as np
from pathlib import Path
from tracking_markers import __version__
from tracking_markers.tracking_points import track_points

test_data_dir = Path(__file__).parents[1] / "data" / "test_data"
test_video_path = test_data_dir / "synthetic_video.mp4"
ground_truth_path = test_data_dir / "ground_truth.npy"


def test_version():
    assert __version__ == '0.9.0'


def generate_synthetic_data():
    """Generates a synthetic video of a moving square and saves ground truth markers."""
    test_data_dir.mkdir(parents=True, exist_ok=True)
    # Always regenerate for now to ensure correctness
    if test_video_path.exists():
        test_video_path.unlink()
    if ground_truth_path.exists():
        ground_truth_path.unlink()

    width, height = 400, 400
    fps = 10
    duration_sec = 4
    num_frames = fps * duration_sec
    radius = 30.0
    # Use center position for easier star drawing
    start_pos = np.array([100.0, 100.0])
    velocity = np.array([4.0, 2.0])

    # Ground truth: shape (num_frames, 5, 2) -> 5 tips
    ground_truth_markers = np.zeros((num_frames, 5, 2))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(test_video_path), fourcc, fps, (width, height), isColor=False)

    for i in range(num_frames):
        # Create white background
        frame = np.ones((height, width), dtype=np.uint8) * 255

        current_pos = start_pos + velocity * i
        cx, cy = current_pos[0], current_pos[1]

        # Calculate star points
        tips = []
        pts = []
        for k in range(5):
            # Outer vertices (tips)
            theta = -np.pi/2 + k * 2*np.pi/5
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            tips.append([x, y])
            pts.append([x, y])

            # Inner vertices
            phi = theta + np.pi/5
            r_in = radius * 0.382
            xi = cx + r_in * np.cos(phi)
            yi = cy + r_in * np.sin(phi)
            pts.append([xi, yi])

        # Draw black star
        cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], 0)

        # Ground truth
        ground_truth_frame = np.array(tips)
        # Transform to FLIPPED coordinates: y' = height - 1 - y
        ground_truth_frame[:, 1] = height - 1 - ground_truth_frame[:, 1]

        ground_truth_markers[i] = ground_truth_frame
        out.write(frame)

    out.release()
    np.save(ground_truth_path, ground_truth_markers)


def test_track_points():
    # Ensure data exists. regenerate if missing
    if not test_video_path.exists() or not ground_truth_path.exists():
        generate_synthetic_data()

    # Load ground truth
    ground_truth = np.load(ground_truth_path)
    initial_markers = ground_truth[0]

    # Run tracking
    tracked_markers = track_points(
        video_path=test_video_path,
        markers=initial_markers,
        step_size=1,
        search_window_size=40,
        marker_template_size=20,
        show_progress_bar=False,
        show_tracked_frame=False,
        show_tracked_box=True,
        save_animation_path=test_data_dir / "tracked_video.mp4"
    )

    # Inspect shapes
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Tracked result shape: {tracked_markers.shape}")

    # Check if we tracked all frames
    assert tracked_markers.shape == ground_truth.shape

    # Check accuracy
    error = np.linalg.norm(tracked_markers - ground_truth, axis=2)
    mean_error = np.mean(error)
    max_error = np.max(error)

    print(f"Mean error: {mean_error}")
    print(f"Max error: {max_error}")

    # Allow reasonable error.
    assert mean_error < 1.0, f"Mean tracking error too high: {mean_error}"
    assert max_error < 5.0, f"Max tracking error too high: {max_error}"
