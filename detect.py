import io
import picamera
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, Response

app = Flask(__name__)

interpreter = tflite.Interpreter(model_path='movenet.tflite')
interpreter.allocate_tensors()

previous_keypoints = None

def calculate_speed(previous_keypoints, current_keypoints):
    if previous_keypoints is None:
        return 0, current_keypoints

    ignore_indices = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    mask = np.ones(previous_keypoints.shape[0], dtype=bool)
    mask[ignore_indices] = False

    previous_keypoints = previous_keypoints[mask]
    current_keypoints = current_keypoints[mask]

    displacement = np.sqrt(np.sum((current_keypoints - previous_keypoints) ** 2, axis=1))
    mean_displacement = np.mean(displacement)

    return mean_displacement, current_keypoints

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30

def generate_frames():
    global previous_keypoints

    stream = io.BytesIO()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        stream.truncate()
        camera.capture(stream, format='jpeg')

        stream.seek(0)
        image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        image = tf.image.decode_jpeg(image, channels=3)  # 수정된 부분: TensorFlow 함수 사용
        image = tf.image.resize(image, [input_shape[1], input_shape[2]])
        image = np.expand_dims(image, axis=0)
        image = (image.astype(np.float32) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        included_keypoints = keypoints_with_scores[0][:, :2]
        speed, previous_keypoints = calculate_speed(previous_keypoints, included_keypoints)

        if speed > 0:
            message = "someone is coming"
        else:
            message = "nobody"

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + stream.getvalue() + b'\r\n\r\n')
        yield (message.encode() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
