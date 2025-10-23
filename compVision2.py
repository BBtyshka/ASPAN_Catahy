import cv2
import mediapipe as mp
import numpy as np
import time

target_points = []  # Changed to list to store multiple points

file_name = 'plane.MOV'

# 3D model points for key facial landmarks (in mm, relative to face center)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float32)

# Distortion coefficients (assume no lens distortion for simplicity)
dist_coeffs = np.zeros((4, 1))

def get_camera_matrix(image_shape):
    """Approximate camera matrix from image size (focal length ~= width)."""
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    return camera_matrix

def get_2d_points(face_landmarks, image_shape):
    """Extract 2D points corresponding to the 3D model points."""
    h, w = image_shape[:2]
    # MediaPipe indices corresponding to the 3D model points:
    # 1: Nose tip, 152: Chin, 33: Left eye outer, 263: Right eye outer
    # 61: Left mouth corner, 291: Right mouth corner
    landmark_indices = [1, 152, 33, 263, 61, 291]
    image_points = []
    for idx in landmark_indices:
        landmark = face_landmarks.landmark[idx]
        x = landmark.x * w
        y = landmark.y * h
        image_points.append([x, y])
    return np.array(image_points, dtype=np.float32)

def point_on_extended_line(target_point, nose_start, nose_end, threshold=30):
    """Check if target point is near the nose direction line AND in front of the face."""
    target = np.array(target_point, dtype=np.float32)
    start = np.array(nose_start, dtype=np.float32)
    end = np.array(nose_end, dtype=np.float32)

    direction = end - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return False, float('inf')

    direction = direction / direction_norm
    to_target = target - start
    projection_length = np.dot(to_target, direction)
    
    # Check if the point is in front (positive projection) or behind (negative projection)
    if projection_length < 0:
        # Point is behind the nose direction - not intersecting
        return False, float('inf')
    
    projection_point = start + projection_length * direction
    distance = np.linalg.norm(target - projection_point)
    return distance <= threshold, distance

def draw_pose(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    """Draw 3D axes, nose direction, and target intersection info."""
    axes_points = np.array([
        (0, 0, 0),      # Origin
        (100, 0, 0),    # X axis (red)
        (0, 100, 0),    # Y axis (green)
        (0, 0, -100)    # Z axis (blue, negative Z is forward)
    ], dtype=np.float32)

    axes_2d, _ = cv2.projectPoints(axes_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Nose direction: start at origin and extend forward
    nose_line_points = np.array([
        (0, 0, 0),
        (0, 0, 1000)   # far-forward point in model coordinates
    ], dtype=np.float32)

    nose_line_2d, _ = cv2.projectPoints(nose_line_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    axes_2d = np.int32(axes_2d).reshape(-1, 2)
    nose_line_2d = np.int32(nose_line_2d).reshape(-1, 2)

    origin = tuple(axes_2d[0])
    cv2.arrowedLine(image, origin, tuple(axes_2d[1]), (0, 0, 255), 3)  # X - Red
    cv2.arrowedLine(image, origin, tuple(axes_2d[2]), (0, 255, 0), 3)  # Y - Green
    cv2.arrowedLine(image, origin, tuple(axes_2d[3]), (255, 0, 0), 3)  # Z - Blue

    nose_start = tuple(nose_line_2d[0])
    nose_end = tuple(nose_line_2d[1])

    # Draw nose direction line
    cv2.arrowedLine(image, nose_start, nose_end, (0, 255, 255), 4)

    # Check all target points against the nose line
    if target_points:
        y_offset = 60
        any_intersecting = False
        for idx, target_point in enumerate(target_points):
            is_intersecting, distance = point_on_extended_line(target_point, nose_start, nose_end, threshold=50)
            color = (0, 255, 0) if is_intersecting else (0, 0, 255)
            
            # Draw circle with number
            cv2.circle(image, target_point, 12, color, -1)
            cv2.putText(image, f"{idx+1}", (target_point[0] - 5, target_point[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display distance info
            cv2.putText(image, f"Point {idx+1}: {distance:.1f}px", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
            if is_intersecting:
                any_intersecting = True
                print(f'Point {idx+1} Intersect')
        
        if any_intersecting:
            cv2.putText(image, "INTERSECTING!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display a simple pose indicator (approx. rotation vector magnitude)
    try:
        pitch_deg = np.degrees(np.linalg.norm(rotation_vector))
        cv2.putText(image, f"Pose vec mag: {pitch_deg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception:
        pass

def mouse_callback(event, x, y, flags, param):
    global target_points
    if event == cv2.EVENT_LBUTTONDOWN:
        target_points.append((x, y))
        print(f"Target point {len(target_points)} added at: {(x, y)}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to remove the last point
        if target_points:
            removed = target_points.pop()
            print(f"Removed last point: {removed}")
        else:
            print("No points to remove")

mp_drawing = mp.solutions.drawing_utils
thin_landmark_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
thin_connection_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def open_capture(source):
    """Open a cv2.VideoCapture from either integer camera index or file path."""
    try:
        # try integer index
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(source)
    return cap

def main():
    # Start with video file, can toggle to webcam
    use_webcam = False
    source = file_name if not use_webcam else 0

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = 30.0   # fallback if FPS is unknown
    frame_delay_ms = int(round(1000.0 / source_fps))
    print(f"Source FPS: {source_fps:.2f}, frame_delay_ms={frame_delay_ms}")
    print(f"Current source: {'Webcam' if use_webcam else 'Video file'}")
    print("Press 'w' for webcam, 'v' for video file, 'q' or ESC to quit")
    print("Left-click to add target points, right-click to remove last point")
    print("Press 'c' to clear all points")

    window_name = 'MediaPipe Face Mesh with Pose'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            frame_start = time.time()
            ret, image = cap.read()
            if not ret:
                # End of video file or camera error -> exit the loop
                print("End of stream or cannot fetch frame. Exiting.")
                break
            

            # Convert to RGB for MediaPipe
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            image.flags.writeable = True
            annotated = image.copy()

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        camera_matrix = get_camera_matrix(image.shape)
                        image_points = get_2d_points(face_landmarks, image.shape)
                        # Solve PnP (returns success, rvec, tvec)
                        pnp_result = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
                        if pnp_result is not None:
                            # Unpack robustly (OpenCV versions differ)
                            if len(pnp_result) == 3:
                                success, rotation_vector, translation_vector = pnp_result
                            else:
                                # fallback
                                rotation_vector, translation_vector = pnp_result
                                success = True
                            if success:
                                draw_pose(annotated, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    except Exception as e:
                        # If PnP failed, continue drawing mesh so user can see landmarks
                        print("PnP or pose draw error:", e)
                        pass

                    # Draw face contours for reference
                    mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=thin_landmark_spec,
                        connection_drawing_spec=thin_connection_spec
                    )

            # Show the annotated frame (do not flip for video clarity)
            cv2.imshow(window_name, annotated)

            elapsed_ms = (time.time() - frame_start) * 1000.0
            wait_ms = max(1, int(frame_delay_ms - elapsed_ms))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):  # Clear all points
                target_points.clear()
                print("All target points cleared")
            elif key == ord('w'):  # Switch to webcam
                if not use_webcam:
                    use_webcam = True
                    cap.release()
                    cap = open_capture(0)
                    if cap.isOpened():
                        source_fps = cap.get(cv2.CAP_PROP_FPS)
                        if source_fps is None or source_fps <= 0:
                            source_fps = 30.0
                        frame_delay_ms = int(round(1000.0 / source_fps))
                        print(f"Switched to WEBCAM - FPS: {source_fps:.2f}")
                    else:
                        print("ERROR: Could not open webcam")
                        use_webcam = False
                        cap = open_capture(file_name)
            elif key == ord('v'):  # Switch to video file
                if use_webcam:
                    use_webcam = False
                    cap.release()
                    cap = open_capture(file_name)
                    if cap.isOpened():
                        source_fps = cap.get(cv2.CAP_PROP_FPS)
                        if source_fps is None or source_fps <= 0:
                            source_fps = 30.0
                        frame_delay_ms = int(round(1000.0 / source_fps))
                        print(f"Switched to VIDEO FILE - FPS: {source_fps:.2f}")
                    else:
                        print("ERROR: Could not open video file")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()