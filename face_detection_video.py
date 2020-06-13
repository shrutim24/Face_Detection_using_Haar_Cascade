import cv2
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
videocap = cv2.VideoCapture(0)

while True:
    retval, img = videocap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Face Detection-Video', img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

videocap.release()
cv2.destroyAllWindows()