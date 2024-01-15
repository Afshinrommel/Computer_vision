import numpy as np
import cv2
import face_detection
# from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet import preprocess_input
from keras.models import load_model
from configs import config
from euc import eucli_dist
from draw_rectangle import draw




def detect_face_mask(net,BASE_PATH,img,frame,classes,output_layers,width):   
       
    height, width, channels = img.shape
    # Detect Objects in the Frame with YOLOv3
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    mask_classifier = load_model("Models/ResNet50_Classifier.h5")
    class_ids = []
    confidences = []
    boxes = []
    person_coordinates = []
# Store Detected Objects with Labels, Bounding_Boxes and their Confidences
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get Center, Height and Width of the Box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Topleft Co-ordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, config.MIN_CONF, config.NMS_THRESH)
    # Initialize empty lists for storing Bounding Boxes of People and their Faces
    persons = []
    masked_faces = []
    unmasked_faces = []
    # Work on Detected Persons in the Frame

    for i in range(len(boxes)):
     if i in indexes:
        box = np.array(boxes[i])
        box = np.where(box<0,0,box)
        (x, y, w, h) = box
        label = str(classes[class_ids[i]])
        if label=='person':
         persons.append([x,y,w,h])
    person_count = len(persons)
    violate = eucli_dist(persons)

    persons = []
    for i in range(len(boxes)):
        if i in indexes:
            box = np.array(boxes[i])
            box = np.where(box<0,0,box)
            (x, y, w, h) = box
            label = str(classes[class_ids[i]])
            if label=='person':
                persons.append([x,y,w,h])
                # Save Image of Cropped Person (If not required, comment the command below)
                cv2.imwrite(BASE_PATH + "Results/Extracted_Persons/"+str(frame)
                            +"_"+str(len(persons))+".jpg",
                            img[y:y+h,x:x+w])
                # Detect Face in the Person
                person_rgb = img[y:y+h,x:x+w,::-1]   # Crop & BGR to RGB
                detections = detector.detect(person_rgb)
                # If a Face is Detected
                if detections.shape[0] > 0:
                  detection = np.array(detections[0])
                  detection = np.where(detection<0,0,detection)
                  # Calculating Co-ordinates of the Detected Face
                  x1 = x + int(detection[0])
                  x2 = x + int(detection[2])
                  y1 = y + int(detection[1])
                  y2 = y + int(detection[3])
                  try :
                    # Crop & BGR to RGB
                    face_rgb = img[y1:y2,x1:x2,::-1]   

                    # Preprocess the Image
                    face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    face_arr = preprocess_input(face_arr)

                    # Predict if the Face is Masked or Not
                    score = mask_classifier.predict(face_arr)

                    # Determine and store Results
                    if score[0][0]<0.5:
                      masked_faces.append([x1,y1,x2,y2])
                    else:
                      unmasked_faces.append([x1,y1,x2,y2])
                    # Save Image of Cropped Face (If not required, comment the command below)
                    cv2.imwrite(BASE_PATH + "Results/Extracted_Faces/"+str(frame)
                                +"_"+str(len(persons))+".jpg",
                                img[y1:y2,x1:x2])
                  except:
                    continue            
    draw(persons,violate,img,frame,masked_faces,unmasked_faces,BASE_PATH,width)
    return(masked_faces,unmasked_faces)