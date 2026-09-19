from imageai.Detection import ObjectDetection
import os
import cv2


execution_path = os.path.dirname(os.path.abspath(__file__))
# print(execution_path)
# print(os.path.join(execution_path , "x2.jpg"))

# image=cv2.imread(os.path.join(execution_path , "x2.jpg"))
# cv2.imshow('abd',image)
# cv2.waitKey(0)
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "model.h5"))

detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "x2.jpg"), output_image_path=os.path.join(execution_path , "image2.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )