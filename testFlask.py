from flask import Flask, request, jsonify
import face_recognition
import os
import numpy as np
import cv2

app = Flask(__name__)

known_encodings = []
known_names = []

def load_known_faces():
    folder = 'Faces'
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f'✅ Đã tải: {name}')
            else:
                print(f'⚠️ Không nhận diện được mặt trong: {filename}')

load_known_faces()

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400

    file = request.files['photo']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    scale = 0.25
    small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_img)
    face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

    results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        results.append({
            'name': name,
            'top': int(top / scale),
            'right': int(right / scale),
            'bottom': int(bottom / scale),
            'left': int(left / scale),
        })

    return jsonify({
        'count': len(results),
        'faces': results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
