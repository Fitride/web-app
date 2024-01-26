from flask import Flask, render_template, Response, redirect
import cv2
from modules.angles import Angles
import mediapipe as mp


app = Flask(__name__)
camera = None  # Initialize camera as None
is_camera_active = False  # Initially, the camera is not active


def generate_frames():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    global is_camera_active

    print("Camera is active: ", is_camera_active)
    while is_camera_active:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw pose skeleton
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            mid_hip = [(hip[0] + hip[0]) / 2, (hip[1] + hip[1]) / 2]

            # Calculer les angles
            angle_acromion_coude_poignet = Angles.calculate_angle(
                shoulder, elbow, wrist)
            angle_hanche_genou_cheville = Angles.calculate_angle(
                hip, knee, ankle)
            angle_bassin_acromion_poignet = Angles.calculate_angle(
                hip, shoulder, wrist)
            angle_acromion_bassin_cheville = Angles.calculate_angle(
                shoulder, hip, ankle)
            angle_dos = Angles.calculate_angle(hip, mid_hip, shoulder)

            """
            ## Mettre à jour les valeurs des angles
            angle_texts[0].text(f"Angle bras: {int(angle_acromion_coude_poignet)}")
            angle_texts[1].text(f"Angle jambe: {int(angle_hanche_genou_cheville)}")
            angle_texts[2].text(f"Angle Bras/buste: {int(angle_bassin_acromion_poignet)}")
            angle_texts[3].text(f"Angle tronc: {int(angle_acromion_bassin_cheville)}")
            angle_texts[4].text(f"Angle dos: {int(angle_dos)}")
            """
        # Update Streamlit UI with the processed frame

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        # Check if the "Use Webcam" button has been dis
        if _:
            frame = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

    print("Camera is not active: ", is_camera_active)
    camera.release()


def initialize_camera():
    global camera, is_camera_active
    # Release the camera if it's open
    if camera is not None and camera.isOpened():
        camera.release()

    # Re-initialize the camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not initialize camera")

    # Indicate that the camera is now active
    is_camera_active = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    initialize_camera()  # Ensure the camera is initialized
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/shutdown_video', methods=['GET'])
def shutdown_video():
    global is_camera_active
    # Signal to stop/run the video capture loop
    if (is_camera_active == False):
        is_camera_active = True
        return {'status': 'Camera on'}
    else:
        is_camera_active = False
        return {'status': 'Camera shutdown'}


if __name__ == "__main__":
    app.run(debug=True)
