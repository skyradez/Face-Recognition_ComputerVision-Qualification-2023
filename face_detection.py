import cv2 as cv

classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv.imread('assets/slc_222.jpg')

width = 1020
ratio = width / image.shape[1]
height = int(image.shape[0] * ratio)
new_image = cv.resize(image, (width, height))

gray = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(gray)

for (top, right, bottom, left) in faces:
    cv.rectangle(new_image, (top, right), (top + bottom, right + left), (0, 0, 255), 2)

    face = new_image[right:right + left, top:top + bottom]
    face = cv.GaussianBlur(face, (23, 23), 30)

    new_image[right:right + face.shape[0], top:top + face.shape[1]] = face

cv.imshow('Face Detection', new_image)
cv.waitKey()
