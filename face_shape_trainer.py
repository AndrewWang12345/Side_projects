import cv2
import mediapipe as mp
import numpy
import matplotlib.pyplot as plt
from sklearn import neighbors
from pathlib import Path
from joblib import dump, load
import array
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
neighbor = neighbors.KNeighborsClassifier(n_neighbors=1)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    pictures = list(Path("face_shape1").iterdir())
    coordinates = []
    person_size = []
    for picture in pictures:
        print(str(picture))
        frame = cv2.imread(str(picture))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        if results == None:
            continue
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
        image_scale = image.shape[1]/image.shape[0]
        scale = 1 / numpy.max(landmarks_y)
        landmarks_x = (landmarks_x) * scale * image_scale
        landmarks_y = (landmarks_y) * scale
        landmarks_z = (landmarks_z) * scale * image_scale
        coordinate = [1 for x in range(1404)]
        for i in range(1404):
            if i < 468:
                coordinate[i] = int(landmarks_x[i] * 1000000000000000)
            elif 468 <= i and i < 936:
                coordinate[i] = int(landmarks_y[i - 468] * 1000000000000000)
            else:
                coordinate[i] = int(landmarks_z[i - 936] * 1000000000000000)
        #print(coordinate)
        coordinates.append(coordinate)
        person_size.append(2)
        print("here")

    pictures = list(Path('face_shape2').iterdir())
    for picture in pictures:
        frame = cv2.imread(str(picture))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        if results == None:
            continue
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
        landmarks_x = numpy.array(landmarks_x) * scale * image_scale
        landmarks_y = numpy.array(landmarks_y) * scale
        landmarks_z = numpy.array(landmarks_z) * scale * image_scale
        coordinate = [1 for x in range(1404)]
        for i in range(1404):
            if i < 468:
                coordinate[i] = int(landmarks_x[i] * 1000000000000000)
            elif 468 <= i and i < 936:
                coordinate[i] = int(landmarks_y[i-468] * 1000000000000000)
            else:
                coordinate[i] = int(landmarks_z[i-936] * 1000000000000000)
        coordinates.append(coordinate)
        person_size.append(1)
    neighbor.fit(coordinates, person_size)
    dump(neighbor, 'face_shape.joblib')
    print(coordinates)
