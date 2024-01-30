from flask import Flask, render_template, Response
from modules.drawing import Drawing

app = Flask(__name__)
camera = None  # Initialize camera as None
is_camera_active = False  # Initially, the camera is not active


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    draw_utils = Drawing()  # Initialize Drawing class
    draw_utils.initialize_camera()  # Ensure the camera is initialized
    return Response(draw_utils.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
