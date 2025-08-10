from flask import Flask, render_template, Response, request, jsonify
import os
import cv2
import time
from werkzeug.utils import secure_filename
from deepface import DeepFace
import datetime
from gtts import gTTS
import pygame

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MATCH_FOLDER = 'matches'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCH_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DB_FOLDER = 'known_criminals'
detections = []  # Store detections for table

pygame.mixer.init()

# ------------------------------
# Webcam Face Recognition
# ------------------------------
def generate_webcam():
    cap = cv2.VideoCapture(0)
    detected = set()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_path = "temp_webcam_frame.jpg"
        cv2.imwrite(frame_path, frame)

        try:
            results = DeepFace.find(
                img_path=frame_path,
                db_path=DB_FOLDER,
                model_name='ArcFace',
                detector_backend='mtcnn',
                enforce_detection=False
            )

            if len(results) > 0 and not results[0].empty:
                top_match = results[0].iloc[0]
                identity = os.path.basename(top_match['identity'])
                label = os.path.splitext(identity)[0]
                distance = float(top_match['distance'])
                confidence = (1 - distance) * 100
                confidence_str = f"{confidence:.2f}"
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                cv2.putText(frame, f"{label} ({confidence_str}%) @ {timestamp}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if label not in detected:
                    detected.add(label)
                    tts = gTTS(text=f"{label} detected with {confidence_str} percent confidence", lang='en')
                    audio_path = f"{label}_webcam.mp3"
                    tts.save(audio_path)
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()

                detections.append({
                    "name": label,
                    "confidence": confidence_str,
                    "time": timestamp
                })
                if len(detections) > 100:
                    detections.pop(0)

            else:
                cv2.putText(frame, "No Match", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            print(f"Webcam Error: {e}")
            cv2.putText(frame, "Error", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1)

    cap.release()


# ------------------------------
# Uploaded Video Stream
# ------------------------------
def generate_uploaded_video_stream(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_num = 0
    detected = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = "frame_temp.jpg"
        cv2.imwrite(frame_path, frame)

        try:
            results = DeepFace.find(
                img_path=frame_path,
                db_path=DB_FOLDER,
                model_name='ArcFace',
                detector_backend='mtcnn',
                enforce_detection=False
            )

            if len(results) > 0 and not results[0].empty:
                top_match = results[0].iloc[0]
                identity = os.path.basename(top_match['identity'])
                label = os.path.splitext(identity)[0]
                distance = float(top_match['distance'])
                confidence = (1 - distance) * 100
                confidence_str = f"{confidence:.2f}"

                # ✅ Get video playback time (not PC time)
                video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                timestamp = time.strftime('%H:%M:%S', time.gmtime(video_time_sec))

                cv2.putText(frame, f"{label} ({confidence_str}%) @ {timestamp}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                shot_name = f"match_{label}_frame{frame_num}_{timestamp.replace(':','-')}.jpg"
                cv2.imwrite(os.path.join(MATCH_FOLDER, shot_name), frame)

                if label not in detected:
                    detected.add(label)
                    tts = gTTS(text=f"Match found: {label} with {confidence_str} percent confidence.", lang='en')
                    audio_path = f"match_{label}.mp3"
                    tts.save(audio_path)
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()

                detections.append({
                    "name": label,
                    "confidence": confidence_str,
                    "time": timestamp
                })
                if len(detections) > 100:
                    detections.pop(0)

            else:
                cv2.putText(frame, "No Match", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            print(f"Video Error: {e}")
            cv2.putText(frame, "Error", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_num += 1
        time.sleep(1)

    cap.release()


# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploaded_video_feed/<filename>')
def uploaded_video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_uploaded_video_stream(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No file uploaded."

    file = request.files['video']
    if file.filename == '':
        return "Empty file."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return render_template('stream_video.html', filename=filename)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('name', '').lower()
    matches = []
    for file in os.listdir(DB_FOLDER):
        if query in file.lower():
            matches.append(file)
    return render_template('search_results.html', results=matches, query=query)

@app.route('/detections')
def get_detections():
    return jsonify(detections)

@app.route('/clear_detections', methods=['POST'])
def clear_detections():
    detections.clear()
    return jsonify({"status": "cleared"})

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
