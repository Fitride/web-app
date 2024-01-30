
import cv2
import mediapipe as mp
import numpy as np
try:
    from angles import Angles
except:
    from .angles import Angles

class Drawing:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.custom_pose_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
        ]

    @staticmethod
    def initialize_camera(camera) -> int:
        """
            Function to Shutdown or Setup the camera.
            Returns:
                int: 0 if function run correctly
        """
        # Release the camera if it's open
        if camera is not None and camera.isOpened():
            camera.release()

        # Re-initialize the camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Could not initialize camera")

        # Indicate that the camera is now active
        return True, camera
    
    @staticmethod
    def coordonnate_association(landmarks, mp_pose):
        """ 
            Function to associate prediction with body part
            Args:
                landmarks: list of landmarks
                mp_pose: mediapipe pose class 
            Returns:
                list of landmarks (shoulder, elbow, wrist, knee, ankle, hip)
        """
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        return shoulder, elbow, wrist, knee, ankle, hip
    

    def generate_frames(self, is_camera_active, camera):
        """
            Function to display computer vision solution 
            and draw the skeleton on the screen.
            Returns:
                None
        """
        while is_camera_active:
            # read the camera frame
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Failed to read camera frame")

            h, w, c = frame.shape
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame)
            
            frame.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                for connection in self.custom_pose_connections:
                    start_landmark = connection[0]
                    end_landmark = connection[1]
                    start_point = (int(landmarks[start_landmark].x * w), int(landmarks[start_landmark].y * h))
                    end_point = (int(landmarks[end_landmark].x * w), int(landmarks[end_landmark].y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 3) 

                shoulder, elbow, wrist, knee, ankle, hip = Drawing.coordonnate_association(landmarks, self.mp_pose)

                horizontal = [0, hip[1]]  # Ceci crée une ligne horizontale au niveau de la hanche

                # Calculer les angles
                angle_acromion_coude_poignet = Angles.calculate_angle(shoulder, elbow, wrist)
                angle_hanche_genou_cheville = Angles.calculate_angle(hip, knee, ankle)
                angle_bassin_acromion_poignet = Angles.calculate_angle(hip, shoulder, wrist)
                angle_acromion_bassin_cheville = Angles.calculate_angle(shoulder, hip, ankle)
                angle_dos = Angles.calculate_angle(hip, horizontal, shoulder)

                shoulder = (int(shoulder[0] * w), int(shoulder[1] * h))
                elbow = (int(elbow[0] * w), int(elbow[1] * h))
                wrist = (int(wrist[0] * w), int(wrist[1] * h))
                knee = (int(knee[0] * w), int(knee[1] * h))
                ankle = (int(ankle[0] * w), int(ankle[1] * h))
                hip = (int(hip[0] * w), int(hip[1] * h))
                horizontal = (int(horizontal[0] * w), int(horizontal[1] * h))

                # Dessinez les ellipses pour chaque angle calculé
                Drawing.draw_arc(frame, elbow, shoulder, wrist, angle_acromion_coude_poignet, color=(255, 0, 0, 128), transparency=0.5)
                Drawing.draw_arc(frame, knee, ankle, hip, angle_hanche_genou_cheville, color=(0, 255, 0), transparency=0.5)
                Drawing.draw_arc(frame, shoulder, wrist, hip, angle_bassin_acromion_poignet, color=(0, 0, 255), transparency=0.5)
                Drawing.draw_arc(frame, hip, shoulder, ankle, angle_acromion_bassin_cheville, color=(255, 255, 0), transparency=0.5)

                """
                ## Mettre à jour les valeurs des angles
                angle_texts[0].text(f"Angle bras: {int(angle_acromion_coude_poignet)}")
                angle_texts[1].text(f"Angle jambe: {int(angle_hanche_genou_cheville)}")
                angle_texts[2].text(f"Angle Bras/buste: {int(angle_bassin_acromion_poignet)}")
                angle_texts[3].text(f"Angle tronc: {int(angle_acromion_bassin_cheville)}")
                angle_texts[4].text(f"Angle dos: {int(angle_dos)}")
                """

                # Update UI with the processed frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

            # Check if the "Use Webcam" button has been dis
            if _:
                frame = buffer.tobytes()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )

        # Release the camera and close the window if is_camera_active == False
        camera.release()

    @staticmethod
    def draw_arc(frame: np.ndarray, center: tuple, start_point: tuple, end_point: tuple, angle: float, color: tuple, transparency: float = 0.5) -> None:
        """
            Draw arc corresponding to the angle.
            Args:
                frame (numpy.ndarray): The frame to draw on.
                center (tuple): The center of the arc.
                start_point (tuple): The start point of the arc.
                end_point (tuple): The end point of the arc.
                angle (float): The angle of the arc.
                color (tuple): The color of the arc.
                transparency (float): The transparency of the arc.
            Returns:
                None
        """
        overlay = frame.copy()
        vector_color = (0, 0, 255)
        vector_scale = 0.5
        dot_radius = 3

        # Convert points into numpy vectors
        center_np = np.array(center)
        start_point_np = np.array(start_point)
        end_point_np = np.array(end_point)

        # Calculate the vectors
        vec_start = start_point_np - center_np
        vec_end = end_point_np - center_np

        # Scale down the vectors
        vec_start = vec_start * vector_scale
        vec_end = vec_end * vector_scale

        # Calculate new start and end points for the shorter vectors
        start_point_short = center_np + vec_start
        end_point_short = center_np + vec_end

        # Calculate axes length as the longest distance to the center
        axes_length = (int(np.linalg.norm(vec_start) / 2), int(np.linalg.norm(vec_end) / 2))

        # Calculate starting and ending angles in frame coordinate space
        angle_start = np.degrees(np.arctan2(vec_start[1], vec_start[0])) % 360
        angle_end = angle_start + angle

        angle_start = angle_start % 360
        angle_end = angle_end % 360

        cv2.ellipse(overlay, tuple(center), axes_length, 0, angle_start, angle_end, color, thickness=-1)

        cv2.line(overlay, tuple(center), tuple(start_point_short.astype(int)), vector_color, thickness=2)
        cv2.line(overlay, tuple(center), tuple(end_point_short.astype(int)), vector_color, thickness=2)

        cv2.circle(overlay, tuple(start_point_short.astype(int)), dot_radius, vector_color, thickness=-1)
        cv2.circle(overlay, tuple(end_point_short.astype(int)), dot_radius, vector_color, thickness=-1)

        cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

        return frame
