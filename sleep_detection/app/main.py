from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
import time

app = Flask(__name__)

# Configuración de Mediapipe
mp_face_mesh = mp.solutions.face_mesh
INDEX_LEFT_EYE = [33, 160, 158, 133, 153, 144]
INDEX_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.26
MICROSLEEP_FRAMES = 3 * 30  # Microsueños: 3 segundos a 30 FPS

class State:
    """Clase para manejar el estado de detección de microsueños."""
    def __init__(self):
        self.microsleep_counter = 0
        self.aux_counter = 0
        self.beep_active = False
        self.last_frame_time = time.time()
        self.is_microsleep = False

    def reset(self):
        """Restablece el estado."""
        self.__init__()

state = State()

def eye_aspect_ratio(coordinates):
    """Calcula la relación de aspecto del ojo (EAR)."""
    d_a = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_b = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_c = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_a + d_b) / (2 * d_c)

def process_eye_landmarks(face_landmarks, indexes, width, height):
    """Obtiene las coordenadas de los ojos a partir de los landmarks."""
    return [[int(face_landmarks.landmark[i].x * width),
             int(face_landmarks.landmark[i].y * height)] for i in indexes]

def detect_microsleep(ear):
    """Detecta microsueños en base al EAR."""
    if ear < EAR_THRESHOLD:
        state.aux_counter += 1
        if state.aux_counter >= MICROSLEEP_FRAMES:
            if not state.beep_active:
                state.microsleep_counter += 1
                state.beep_active = True
            state.is_microsleep = True
    else:
        state.beep_active = False
        state.is_microsleep = False
        state.aux_counter = 0

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Procesa un frame recibido y detecta microsueños."""
    state.last_frame_time = time.time()
    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        results = face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"error": "No face detected"}), 400
        
        face_landmarks = results.multi_face_landmarks[0]
        left_eye = process_eye_landmarks(face_landmarks, INDEX_LEFT_EYE, width, height)
        right_eye = process_eye_landmarks(face_landmarks, INDEX_RIGHT_EYE, width, height)
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        detect_microsleep(ear)
        
        return jsonify({
            "microsleep_counter": state.microsleep_counter,
            "is_microsleep": state.is_microsleep
        })

@app.route('/reset', methods=['POST'])
def reset_state():
    """Restablece el estado de la detección."""
    state.reset()
    return jsonify({"message": "State reset successfully"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
