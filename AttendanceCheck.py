import cv2
import os
import time
import sys

import numpy as np
import PIL
from PIL import Image

if len(sys.argv) != 3:
    print ("Arguments required: path/to/dataset path/to/haarcasecades_frontalface")
    sys.exit(0)
datasetPath = sys.argv[1]
haarcascadesPath = sys.argv[2]

# Face Detection cascade
face_cascade = cv2.CascadeClassifier(haarcascadesPath)
# Face recognizer cascade
recognizer = cv2.face.createLBPHFaceRecognizer()

# Map images to name
map = {}
# Track student attendances
studentsLate = set([])

# retrieve faces from training set
def get_images_and_labels(path_to_data):    
    if not os.path.isdir(path_to_data):
        print ("Dataset folder not found!")
        sys.exit(0)
    
    # find image paths
    image_folders = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data) \
                    if os.path.isdir(os.path.join(path_to_data, f))]
    image_paths = [os.path.join(f, image_path) for f in image_folders \
                    for image_path in os.listdir(f) if image_path.endswith(".png")]
    
    # stores faces detected in image
    images = []
    # stores label of the image
    labels = []
    
    totalFaces = 0
    
    # map every image to a keyID
    for image_path in image_paths:
        # read image as grayscale
        img = cv2.imread(image_path, 0)
        # Get the label
        name = os.path.split(os.path.dirname(image_path))[1][:]
        # map face to ID
        faces = face_cascade.detectMultiScale(img)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            images.append(img[y:y+h,x:x+w])
            label = len(map) - 1
            labels.append(label)
            map[label] = name
            studentsLate.add(name)
            totalFaces += 1
            
            cv2.imshow(name + str(label), img[y:y+h, x:x+w])
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        else:
            print ("Failed to find face at ", image_path)
            os.remove(image_path)
    print("Training total: ", totalFaces)

    return images, labels

if __name__ == '__main__':
    images, labels = get_images_and_labels(datasetPath)
    
    if len(images) == 0 or len(labels) == 0:
        print ("No data!")
        sys.exit(0)
    
    # train recognizer from dataset
    recognizer.train(images, np.array(labels))
    print ("Successfully trained!")
   
    # open camera
    cap = cv2.VideoCapture(0)
    
    # start checking attendance
    #
    # g = take picture of student
    # q or esc to end
    while(1):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if cv2.waitKey(50) == ord('g'):
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Recognize faces
            if len(faces) > 0:
                x,y,w,h = faces[0]
                if h > 100 and w > 100:
                    roi_gray = gray[y:y+h, x:x+w]
                    label_predicted = recognizer.predict(roi_gray)
                    name = map.get(label_predicted)
                    print(name, "marked present!")
    
                    studentsLate.discard(name) # marked present
                    
                    cv2.imshow(map.get(label_predicted), roi_gray)
                    cv2.waitKey(4000)
        
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# display late students at end of attendance checking
for name in studentsLate:
    print ("{} is late!".format(name))


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
