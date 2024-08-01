from flask import Flask, render_template, request, redirect, url_for, flash, Response
import os
import cv2
import face_recognition
import numpy as np
from werkzeug.utils import secure_filename
from pathlib import Path
import csv
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'visages_connus'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

known_face_encodings = []
known_face_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_event(evenement, details=''):
    with open('historique.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), evenement, details])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/historique')
def history():
    events = []
    try:
        with open('historique.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                events.append({"timestamp": row[0], "message": row[1], "details": row[2]})
    except FileNotFoundError:
        pass
    return render_template('historique.html', events=events)

@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['file']
        if file and allowed_file(file.filename):
            extension = file.filename.rsplit('.', 1)[1].lower()
            filename = secure_filename(f"{name}.{extension}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('Visage ajouté avec succès', 'success')
            log_event('Ajout de visage', f'Visage ajouté: {name}')
            return redirect(url_for('confirmation'))
        else:
            flash('Format de fichier non supporté. Veuillez télécharger une image JPG, JPEG ou PNG.', 'danger')
    return render_template('add_face.html')

@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame = easy_face_reco(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def easy_face_reco(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Inconnu"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)
        log_event('Détection de visage', f'Visage détecté: {name}')

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame

def load_known_faces(input_directory):
    for file_ in Path(input_directory).rglob('*'):
        if file_.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                image = face_recognition.load_image_file(file_)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(file_.stem)
            except Exception as e:
                print(f'[ERREUR] Échec de l\'encodage de {file_}: {e}')

if __name__ == '__main__':
    load_known_faces(UPLOAD_FOLDER)
    app.run(debug=True)
