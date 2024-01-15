import cv2
import os
import numpy as np
from intermediate import detect_face_mask
import warnings
import shutil
import tqdm


BASE_PATH ="C:/Users/js/Documents/P3"

warnings.filterwarnings("ignore", category=UserWarning) 
shutil.rmtree(BASE_PATH+"Results/Extracted_Faces", ignore_errors=True)
os.makedirs(BASE_PATH+'Results/Extracted_Faces')

shutil.rmtree(BASE_PATH+"Results/Extracted_Persons", ignore_errors=True)
os.makedirs(BASE_PATH+"Results/Extracted_Persons")


shutil.rmtree(BASE_PATH+"Results/Frames", ignore_errors=True)
os.makedirs(BASE_PATH+"Results/Frames")


vs = cv2.VideoCapture('Mix.mp4')
#labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open("Models/coco.names").read().strip().split("\n")

net = cv2.dnn.readNet("Models/yolov3.weights", "Models/yolov3.cfg")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
get_out_put = net.getUnconnectedOutLayers()
 
# determine only the "output" layer names that we need from YOLO
ln = [layer_names[i - 1] for i in get_out_put]
ret = True
fps = vs.get(cv2.CAP_PROP_FPS)
n_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

for frame_counter in tqdm.tqdm(range(int(n_frames))):    
    ret,frame =  vs.read()
    if ret == False:
       break
    if ret:
        assert not isinstance(frame,type(None)), 'frame not found'
        resultX = detect_face_mask(net,BASE_PATH,frame,frame_counter,LABELS,ln,width) 


cv2.destroyAllWindows()
vs.release()




