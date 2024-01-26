import mediapipe as mp
import numpy as np
import cv2
import tempfile
from modules.angles import Angles

class StreamlitAddon:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

    @st.cache()
    def image_resize(self, image:np.ndarray, width: int=None, height: int=None, inter: int=cv2.INTER_AREA) -> np.ndarray:
        """
            Resize an image.

            Args:
                image (numpy.ndarray): The image to be resized.
                width (int): The width of the resized image.
                height (int): The height of the resized image.
                inter (int): The interpolation method.

            Returns:
                numpy.ndarray: The resized image.
        
        """
        # initialize the dimensions of the image to be resized and grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        return resized
    

    def calculate_angles_in_image(self, image_path: str) -> None:
        """
            Calculate the angles in an image.

            Args:
                image_path (str): The path to the image.

            Returns:
                None
        """
        cap = cv2.VideoCapture(image_path)

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Obtain landmark coordinates
                    shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                    wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                    knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
                    ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate angles
                    angle_acromion_coude_poignet = Angles.calculate_angle(shoulder, elbow, wrist)
                    angle_hanche_genou_cheville = Angles.calculate_angle(hip, knee, ankle)
                    angle_bassin_acromion_poignet = Angles.calculate_angle(hip, shoulder, wrist)
                    angle_acromion_bassin_cheville = Angles.calculate_angle(shoulder, hip, ankle)
                    angle_dos = Angles.calculate_angle(hip, shoulder, [hip[0], (hip[1] + shoulder[1]) / 2])

                    # Display angles
                    print(f"Angle bras: {round(angle_acromion_coude_poignet, 2)}")
                    print(f"Angle jambe: {round(angle_hanche_genou_cheville, 2)}")
                    print(f"Angle Bras/buste: {round(angle_bassin_acromion_poignet, 2)}")
                    print(f"Angle tronc: {round(angle_acromion_bassin_cheville, 2)}")
                    print(f"Angle dos: {round(angle_dos, 2)}")
                    
                    # Draw pose skeleton
                    self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                except Exception as e:
                    print(f"Error (Streamlit Addon): Calcule d angles error -> {e}")

                #st.image(image, channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()

    def run_on_video(self, video_path: str=None) -> None:
        """
            Calculate angles on live video.

            Args:
                video_path (str): The path to the video.

            Returns:
                None
        """
        #use_webcam = st.sidebar.button('Use Webcam')
        #record = st.sidebar.checkbox("Upload a video")
        
        #if record:
        #st.checkbox("Video", value=True)
        """
        st.sidebar.markdown('---')
        sameer=""
        st.markdown(' ## Output')
        st.markdown(sameer)
        """
        stframe = st.empty()
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not video_file_buffer:
            if use_webcam:
                vid = cv2.VideoCapture(0)
            else:
                vid = cv2.VideoCapture(video_path)
                tfflie.name = video_path

        else:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
        #out = cv2.VideoWriter('assets/Videos/temp/output1.mp4', codec, fps_input, (width, height))

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.sidebar.markdown('---')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 400px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 400px;
                margin-left: -400px;
            
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown('### Angles détectés')  # Titre initial pour les angles
        angle_texts = [st.empty() for _ in range(5)]  # Création de 5 éléments vides pour afficher les angles

        while use_webcam:
            _, img = vid.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw pose skeleton
                self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
                ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
                hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]

                mid_hip = [(hip[0] + hip[0]) / 2, (hip[1] + hip[1]) / 2]

                # Calculer les angles
                angle_acromion_coude_poignet = Angles.calculate_angle(shoulder, elbow, wrist)
                angle_hanche_genou_cheville = Angles.calculate_angle(hip, knee, ankle)
                angle_bassin_acromion_poignet = Angles.calculate_angle(hip, shoulder, wrist)
                angle_acromion_bassin_cheville = Angles.calculate_angle(shoulder, hip, ankle)
                angle_dos = Angles.calculate_angle(hip, mid_hip, shoulder)

                ## Mettre à jour les valeurs des angles
                angle_texts[0].text(f"Angle bras: {int(angle_acromion_coude_poignet)}")
                angle_texts[1].text(f"Angle jambe: {int(angle_hanche_genou_cheville)}")
                angle_texts[2].text(f"Angle Bras/buste: {int(angle_bassin_acromion_poignet)}")
                angle_texts[3].text(f"Angle tronc: {int(angle_acromion_bassin_cheville)}")
                angle_texts[4].text(f"Angle dos: {int(angle_dos)}")

            # Update Streamlit UI with the processed frame
                stframe.image(img, channels="BGR")

            # Check if the "Use Webcam" button has been dis
            if not use_webcam:
                break

        vid.release()
        #out.release()