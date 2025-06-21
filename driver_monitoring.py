
import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79 # This threshold might need tuning
MOUTH_OPEN_DURATION_THRESH = 2 # seconds

def get_ear(eye_landmarks):
    # Calculate the Euclidean distances between the two sets of vertical eye landmarks
    # and the horizontal eye landmark.
    p2_p6 = ((eye_landmarks[1].x - eye_landmarks[5].x)**2 + (eye_landmarks[1].y - eye_landmarks[5].y)**2)**0.5
    p3_p5 = ((eye_landmarks[2].x - eye_landmarks[4].x)**2 + (eye_landmarks[2].y - eye_landmarks[4].y)**2)**0.5
    p1_p4 = ((eye_landmarks[0].x - eye_landmarks[3].x)**2 + (eye_landmarks[0].y - eye_landmarks[3].y)**2)**0.5

    # Calculate the Eye Aspect Ratio (EAR)
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

def get_mar(mouth_landmarks):
    # Calculate the Euclidean distances for mouth aspect ratio
    # Landmarks for mouth: A, B, C, D (inner mouth vertical)
    # E, F (outer mouth horizontal)
    # Using indices from a common mouth landmark set (e.g., from Dlib or specific MediaPipe subset)
    # For MediaPipe Face Mesh, we need to pick appropriate landmarks for vertical and horizontal distances.
    # Let's use common indices for inner and outer mouth corners.
    # Inner mouth vertical: 13, 14, 17, 18 (These are relative indices within the selected mouth_landmarks list)
    # Outer mouth horizontal: 78, 308 (These are relative indices within the selected mouth_landmarks list)

    # Corrected indices for MAR calculation based on MediaPipe Face Mesh
    # Vertical: 13, 14 (top lip inner), 17, 18 (bottom lip inner)
    # Horizontal: 78 (left corner), 308 (right corner)

    # The mouth_landmarks list passed to this function will contain the actual landmark objects.
    # We need to select the correct indices from the *original* face_landmarks list to form the mouth_landmarks list.
    # Then, within get_mar, we use relative indices of the *passed* mouth_landmarks list.

    # Let's assume mouth_landmarks contains the following specific MediaPipe indices in order:
    # [78, 308, 13, 14, 17, 18]
    # So, 78 is index 0, 308 is index 1, 13 is index 2, 14 is index 3, 17 is index 4, 18 is index 5

    # Vertical distances (inner mouth)
    # p13_p14 corresponds to landmark 14 and 13 (vertical distance between top and bottom inner lip points)
    # p17_p18 corresponds to landmark 18 and 17 (vertical distance between top and bottom inner lip points)
    # Let's use 13, 14, 17, 18 from the original MediaPipe indices for clarity.
    # Vertical: 13 (upper inner lip), 14 (lower inner lip), 17 (upper inner lip), 18 (lower inner lip)
    # Horizontal: 78 (left mouth corner), 308 (right mouth corner)

    # For MediaPipe, a common set of mouth landmarks for MAR are:
    # A: 13 (upper inner lip, center)
    # B: 14 (lower inner lip, center)
    # C: 78 (left mouth corner)
    # D: 308 (right mouth corner)

    # Let's redefine mouth_landmarks to contain these specific points:
    # [78, 308, 13, 14]
    # Then, in get_mar, we access them by their new relative indices:
    # 78 -> mouth_landmarks[0]
    # 308 -> mouth_landmarks[1]
    # 13 -> mouth_landmarks[2]
    # 14 -> mouth_landmarks[3]

    # Vertical distance: (landmark 14 - landmark 13)
    # Horizontal distance: (landmark 308 - landmark 78)

    # This is a simplified MAR for MediaPipe. More robust MAR uses more points.
    # Let's use the standard 6 points for mouth: 61, 291, 0, 17, 13, 14 (outer and inner mouth corners)
    # Vertical: 13, 14 (inner top/bottom), 17, 0 (outer top/bottom)
    # Horizontal: 61, 291 (left/right corners)

    # Let's use the following indices for MAR calculation:
    # Vertical: 13, 14 (inner mouth top/bottom)
    # Horizontal: 61, 291 (outer mouth left/right)

    # The mouth_landmarks list passed to get_mar should contain these specific landmarks.
    # So, the list should be: [61, 291, 13, 14]
    # Then, inside get_mar:
    # p13_p14 = distance between landmark 13 and 14 (vertical)
    # p61_p291 = distance between landmark 61 and 291 (horizontal)

    # Let's use a common set of 6 points for MAR: A, B, C, D, E, F
    # A: 13 (upper inner lip)
    # B: 14 (lower inner lip)
    # C: 17 (upper outer lip)
    # D: 0 (lower outer lip)
    # E: 61 (left mouth corner)
    # F: 291 (right mouth corner)

    # So, mouth_landmarks will be [61, 291, 0, 13, 14, 17]
    # Then, in get_mar:
    # Vertical distances: (13-14) and (0-17)
    # Horizontal distance: (61-291)

    # Let's use the indices 13, 14, 17, 0 for vertical distances and 61, 291 for horizontal distance.
    # The mouth_landmarks list will be: [61, 291, 0, 13, 14, 17]

    # Vertical distances
    # Distance between 13 and 14 (inner vertical)
    dist_vertical1 = np.linalg.norm(np.array([mouth_landmarks[3].x, mouth_landmarks[3].y]) - np.array([mouth_landmarks[4].x, mouth_landmarks[4].y]))
    # Distance between 0 and 17 (outer vertical)
    dist_vertical2 = np.linalg.norm(np.array([mouth_landmarks[2].x, mouth_landmarks[2].y]) - np.array([mouth_landmarks[5].x, mouth_landmarks[5].y]))

    # Horizontal distance
    # Distance between 61 and 291 (horizontal)
    dist_horizontal = np.linalg.norm(np.array([mouth_landmarks[0].x, mouth_landmarks[0].y]) - np.array([mouth_landmarks[1].x, mouth_landmarks[1].y]))

    mar = (dist_vertical1 + dist_vertical2) / (2.0 * dist_horizontal)
    return mar

def is_hand_on_steering_wheel(hand_landmarks, image_width, image_height):
    # This is a simplified approach. In a real scenario, you\"d calibrate the steering wheel ROI.
    # Assuming a general region for the steering wheel at the bottom center of the frame.
    steering_wheel_roi_x_min = int(image_width * 0.3)
    steering_wheel_roi_x_max = int(image_width * 0.7)
    steering_wheel_roi_y_min = int(image_height * 0.6)
    steering_wheel_roi_y_max = int(image_height * 0.9)

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if steering_wheel_roi_x_min < x < steering_wheel_roi_x_max and \
           steering_wheel_roi_y_min < y < steering_wheel_roi_y_max:
            return True
    return False

def get_head_pose(face_landmarks, image_shape):
    img_h, img_w, img_c = image_shape
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Get the 2D Coordinates
            face_2d.append([x, y])

            # Get the 3D Coordinates
            face_3d.append([x, y, lm.z])

    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_w / 2],
                            [0, focal_length, img_h / 2],
                            [0, 0, 1]])

    # The Distance Matrix
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    return x, y, z

def driver_monitoring():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    yawn_start_time = None
    yawn_detected = False

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False
                results_face_mesh = face_mesh.process(image)
                results_hands = hands.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results_face_mesh.multi_face_landmarks:
                    for face_landmarks in results_face_mesh.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

                        # Eye Aspect Ratio (EAR) calculation
                        left_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                        right_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

                        left_ear = get_ear(left_eye_landmarks)
                        right_ear = get_ear(right_eye_landmarks)

                        avg_ear = (left_ear + right_ear) / 2.0

                        if avg_ear < EYE_AR_THRESH:
                            cv2.putText(image, "EYES CLOSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(image, "EYES OPEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Mouth Aspect Ratio (MAR) calculation for yawning
                        # Corrected mouth landmarks for MAR calculation
                        # Using indices for outer mouth corners (61, 291) and inner mouth points (0, 13, 14, 17)
                        # These indices are based on common MediaPipe Face Mesh examples for MAR.
                        mouth_indices = [61, 291, 0, 13, 14, 17] # Left corner, Right corner, Lower outer, Upper inner, Lower inner, Upper outer
                        mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]
                        mar = get_mar(mouth_landmarks)

                        if mar > MOUTH_AR_THRESH:
                            if yawn_start_time is None:
                                yawn_start_time = time.time()
                            elif (time.time() - yawn_start_time) >= MOUTH_OPEN_DURATION_THRESH:
                                cv2.putText(image, "YAWNING!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                yawn_detected = True
                        else:
                            yawn_start_time = None
                            yawn_detected = False

                        # Head pose estimation for gaze direction
                        x, y, z = get_head_pose(face_landmarks, image.shape)

                        if y < -10:
                            text = "Looking Left"
                        elif y > 10:
                            text = "Looking Right"
                        elif x < -10:
                            text = "Looking Down"
                        elif x > 10:
                            text = "Looking Up"
                        else:
                            text = "Looking Straight"

                        cv2.putText(image, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if results_hands.multi_hand_landmarks:
                    hand_on_wheel = False
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

                        if is_hand_on_steering_wheel(hand_landmarks, image.shape[1], image.shape[0]):
                            hand_on_wheel = True
                            
                    if hand_on_wheel:
                        cv2.putText(image, "HAND ON WHEEL", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "HAND OFF WHEEL", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(image, "NO HAND DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Driver Monitoring", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    driver_monitoring()


