import cv2
import os
import numpy as np

TRAIN_DIR = 'images/train'

train_files = os.listdir(TRAIN_DIR)

train_images = []
train_labels = []
train_names = []

for train_file in train_files:
    if train_file.endswith('.jpg') or train_file.endswith('.jpeg') or train_file.endswith('.png'):
        train_image = cv2.imread(os.path.join(TRAIN_DIR, train_file))
        train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        label = train_file.split(' - ')[0][1:]
        name = train_file.split(' - ')[1].split('.')[0]
        train_images.append(train_image)
        train_labels.append(int(label))
        train_names.append(name)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(train_images, np.array(train_labels))

CONFIDENCE_THRESHOLD = 100

cap = cv2.VideoCapture(0)

if not os.path.exists('output'):
    os.makedirs('output')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(face)

        if confidence < CONFIDENCE_THRESHOLD:
            label_text = f'T{label:03d} - '
            names = [train_names[i] for i in range(len(train_labels)) if train_labels[i] == label]
            if names:
                label_text += ', '.join(names)
            else:
                label_text += 'Unknown'
        else:
            label_text = 'Unknown'

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        photo_saved = False

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]

                label, confidence = face_recognizer.predict(face)

                if confidence < CONFIDENCE_THRESHOLD:
                    label_text = f'T{label:03d} - '
                    names = [train_names[i] for i in range(len(train_labels)) if train_labels[i] == label]
                    if names:
                        label_text += ', '.join(names)
                    else:
                        label_text += 'Unknown'
                else:
                    label_text = 'Unknown'

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                face_roi = frame[y:y + h, x:x + w]
                blur = cv2.GaussianBlur(face_roi, (101, 101), 0)
                frame[y:y + h, x:x + w] = blur

                if not photo_saved:
                    cv2.imwrite(f'output/{label_text}-{int(confidence)}.png', frame)
                    photo_saved = True

                cv2.imshow('Face Recognition', frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or photo_saved:
                    break
cap.release()
cv2.destroyAllWindows()