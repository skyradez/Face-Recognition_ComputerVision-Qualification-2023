import cv2 as cv

image = cv.imread("assets/slc_222.jpg")
width = 1020
ratio = width / image.shape[1]
height = int(image.shape[0] * ratio)
new_image = cv.resize(image, (width, height))

new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

# Using Canny Edge Detection
edges = cv.Canny(image=new_image, threshold1=100, threshold2=200)
cv.imshow('Canny Edge Detection', edges)
cv.waitKey()