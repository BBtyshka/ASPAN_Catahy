import cv2
import mediapipe as mp
import numpy as np

target_point = None

# Add this after your imports
# 3D model points for key facial landmarks (in mm, relative to face center)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float32)

# Camera intrinsic parameters (you should calibrate your specific camera)
# These are approximate values for a typical webcam
def get_camera_matrix(image_shape):
    focal_length = image_shape[1]
    center = (image_shape[1]/2, image_shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    return camera_matrix

# Distortion coefficients (assume no lens distortion for simplicity)
dist_coeffs = np.zeros((4,1))

def get_2d_points(face_landmarks, image_shape):
    """Extract 2D points corresponding to the 3D model points"""
    h, w = image_shape[:2]
    
    # MediaPipe landmark indices for the corresponding 3D points
    # 1: Nose tip, 152: Chin, 33: Left eye corner, 263: Right eye corner
    # 61: Left mouth corner, 291: Right mouth corner
    landmark_indices = [1, 152, 33, 263, 61, 291]
    
    image_points = []
    for idx in landmark_indices:
        landmark = face_landmarks.landmark[idx]
        x = landmark.x * w
        y = landmark.y * h
        image_points.append([x, y])
    
    return np.array(image_points, dtype=np.float32)

def draw_pose(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    """Draw 3D coordinate axes to show pose and check intersections"""
    # 3D axes points
    axes_points = np.array([
        (0, 0, 0),      # Origin
        (100, 0, 0),    # X axis (red)
        (0, 100, 0),    # Y axis (green)  
        (0, 0, -100)    # Z axis (blue)
    ], dtype=np.float32)
    
    # Project 3D points to 2D
    axes_2d, _ = cv2.projectPoints(
        axes_points, 
        rotation_vector, 
        translation_vector, 
        camera_matrix, 
        dist_coeffs
    )

    # Add a forward direction line from nose (extending in negative Z direction)
    nose_line_points = np.array([
        (0, 0, 0),      # Nose tip (origin)
        (0, 0, 1000)   # Point far forward from nose (negative Z for forward)
    ], dtype=np.float32)

    nose_line_2d, _ = cv2.projectPoints(
        nose_line_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )
    
    axes_2d = np.int32(axes_2d).reshape(-1, 2)
    nose_line_2d = np.int32(nose_line_2d).reshape(-1, 2)
    
    # Draw axes
    origin = tuple(axes_2d[0])
    cv2.arrowedLine(image, origin, tuple(axes_2d[1]), (0, 0, 255), 3)  # X - Red
    cv2.arrowedLine(image, origin, tuple(axes_2d[2]), (0, 255, 0), 3)  # Y - Green
    cv2.arrowedLine(image, origin, tuple(axes_2d[3]), (255, 0, 0), 3)  # Z - Blue

    nose_start = tuple(nose_line_2d[0])
    nose_end = tuple(nose_line_2d[1])

    
    # Check intersections and draw target points
    if target_point is not None:
        is_intersecting, distance = point_on_extended_line(target_point, nose_start, nose_end, threshold=50)
        
        # Draw target point
        color = (0, 255, 0) if is_intersecting else (0, 0, 255)
        cv2.circle(image, target_point, 15, color, -1)
        cv2.putText(image, f"Target", (target_point[0]+20, target_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"Distance: {distance:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if is_intersecting:
            cv2.putText(image, "INTERSECTING!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw the nose direction line
    cv2.arrowedLine(image, nose_start, nose_end, (0, 255, 255), 5)  # Yellow thick line
    
    # Display pose information
    pose_text = f"Pitch: {np.degrees(rotation_vector[0][0]):.1f}°"
    cv2.putText(image, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def point_on_extended_line(target_point, nose_start, nose_end, threshold=30):
    """Check if target point is near the extended nose direction line"""
    # Convert to numpy arrays
    target = np.array(target_point, dtype=np.float32)
    start = np.array(nose_start, dtype=np.float32)
    end = np.array(nose_end, dtype=np.float32)
    
    # Line direction vector
    direction = end - start
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm == 0:
        return False, float('inf')
    
    # Normalize direction
    direction = direction / direction_norm
    
    # Vector from line start to target point
    to_target = target - start
    
    # Project target onto the line
    projection_length = np.dot(to_target, direction)
    projection_point = start + projection_length * direction
    
    # Calculate distance from target to projected point
    distance = np.linalg.norm(target - projection_point)
    
    return distance <= threshold, distance

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to set target point"""
    global target_point
    if event == cv2.EVENT_LBUTTONDOWN:
        target_point = (x, y)
        print(f"Target point set to: {target_point}")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
# Replace your main loop with this enhanced version
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Change to 1 for pose estimation
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.namedWindow('MediaPipe Face Mesh with Pose')
        cv2.setMouseCallback('MediaPipe Face Mesh with Pose', mouse_callback)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get camera matrix
                camera_matrix = get_camera_matrix(image.shape)
                
                # Get 2D points
                image_points = get_2d_points(face_landmarks, image.shape)
                
                # Solve PnP
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, 
                    image_points, 
                    camera_matrix, 
                    dist_coeffs
                )
                
                if success:
                    # Draw pose estimation
                    draw_pose(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                
                # Original face mesh drawing (optional)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
        
        cv2.imshow('MediaPipe Face Mesh with Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

