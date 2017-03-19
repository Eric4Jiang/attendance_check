import numpy as np
import cv2
import os
import sys

if len(sys.argv) != 3:
    print("Arguments required: -pata/to/save/images -path/to/haarcascades_frontface")
    sys.exit(0)
datasetPath = sys.argv[1]
haarcascadesPath = sys.argv[2]

# Train face detection software
face_cascade = cv2.CascadeClassifier(haarcascadesPath)
# start camera
cap = cv2.VideoCapture(0)
# image specs
saveDir = datasetPath
ext = ".png"

imNum = 0
nameOfPerson = input("Enter name of person: ")

# make folder for new person
imFolder = saveDir + "/" + nameOfPerson
if not os.path.exists(imFolder):
    os.makedirs(imFolder)

# start taking images of students
while(1):
    #Capture frame-by-frame
    ret, frame = cap.read()
    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # press g to take picture of student
    if cv2.waitKey(30) == ord('g'):
        # name images
        imName = nameOfPerson + str(imNum)
        imPath = saveDir + "/" + nameOfPerson + "/" + imName + ext
        while os.path.isfile(imPath):
            imNum += 1
            imName = nameOfPerson + str(imNum)
            imPath = saveDir + "/" + nameOfPerson + "/" + imName + ext

        # Detect the face in the image
        faces = face_cascade.detectMultiScale(gray)
        if len(faces) > 0:
            cv2.imwrite(imPath, gray)
            cv2.imshow("Face", gray)
            cv2.waitKey(0)
            imNum += 1
    
    # display image being saved
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
cap.release()
cv2.destroyAllWindows()
