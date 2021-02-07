import cv2 as cv
import os

cap = cv.VideoCapture(0) 
face_detector_path = 'cascade_classifier/face_classifier.xml'
if os.path.exists(face_detector_path):
    print('cascade model found')
    faceModel =cv.CascadeClassifier(face_detector_path)
    while True:
        ret, frame =cap.read()
        if not ret:
            print('camera not working')
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        try:
            faces =faceModel.detectMultiScale(
                gray,                               # the b/w image
                scaleFactor = 1.1,                  # adjust the face near and far from camera 
                minNeighbors = 5,                   # for the algo to find face relative to other items
                minSize=(30,30),                    # min box size
                flags=cv.CASCADE_SCALE_IMAGE        # dont care
            )

            for (x,y,w,h) in faces:
                cv.rectangle(frame,(x,y),(x+w,h+y),(0,255,0),5,)
            print(f'found {len(faces)} faces')
        except  Exception as e:
            print(e)
        cv.imshow('thats me',frame)
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()
else:
    print('model path invalid')