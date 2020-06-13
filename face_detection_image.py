import cv2
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('sample.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade_face.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detection(Photo)', img)
cv2.waitKey()
cv2.destroyAllWindows()