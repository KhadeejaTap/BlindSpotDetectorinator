from ultralytics import YOLO
import numpy as np
import threading
import math
import cv2

model = YOLO("yolov8n.pt")

# set variables
WIDTH = 182.88 #CM
FOC_LENGTH = 720 #pixels

# define functions
def calc_dist(r_width,foc_length,pix_width):
    return (r_width * foc_length) / pix_width

# landmark ids if needed

# webcam 
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret: 
        print("Failed to grab frame")
        break # if frame not grabbed, exit loop

    results = model(frame) 
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        pix_width = x2 - x1
        dist = calc_dist(WIDTH, FOC_LENGTH, pix_width)
        if dist <= 400: #limit to a little for than 12ft
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {dist:.2f} cm", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()
# End of file





