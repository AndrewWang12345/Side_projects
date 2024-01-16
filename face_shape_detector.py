import numpy
import cv2
import mediapipe as mp
import joblib
mp_holistic = mp.solutions.holistic
def detect_face(image_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        frame = cv2.imread(image_path)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        landmarks_x = []
        landmarks_y = []
        landmarks_z = []
        for landmark in results.face_landmarks.landmark:
            landmarks_x.append(landmark.x)
            landmarks_y.append(landmark.y)
            landmarks_z.append(landmark.z)
        x_min = numpy.min(landmarks_x)
        y_min = numpy.min(landmarks_y)
        z_min = numpy.min(landmarks_z)
        landmarks_x -= x_min
        landmarks_y -= y_min
        landmarks_z -= z_min
        image_scale = image.shape[1] / image.shape[0]
        scale = 1 / numpy.max(landmarks_y)

        landmarks_x = (landmarks_x) * scale * image_scale
        landmarks_y = ((landmarks_y) * scale)
        landmarks_z = ((landmarks_z) * scale * image_scale)
        coordinate = [1 for x in range(1404)]
        for i in range(1404):
            if i < 468:
                coordinate[i] = int(landmarks_x[i] * 1000000000000000)
            elif 468 <= i and i < 936:
                coordinate[i] = int(landmarks_y[i - 468] * 1000000000000000)
            else:
                coordinate[i] = int(landmarks_z[i - 936] * 1000000000000000)
        neighbor = joblib.load('face_shape.joblib')
        return neighbor.predict([coordinate])