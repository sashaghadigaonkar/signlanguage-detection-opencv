import cv2
import torch
import numpy as np
from ultralytics import YOLO  # Import YOLOv8

# Load YOLOv8 trained model
model = YOLO("runs/runs/classify/train/weights/best.pt")  # Update path if needed

# Load webcam
cap = cv2.VideoCapture(0)

# Labels (Update according to your trained model classes)
label = ["A", "B", "C"]  # Modify based on your dataset

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not access webcam.")
        break

    # Run YOLOv8 classification model
    results = model(img)

    # Extract top predicted class
    for result in results:
        if result.probs is not None:  # Ensure classification result exists
            cls = int(result.probs.top1)  # Get predicted class index
            conf = float(result.probs.top1conf)  # Get confidence score
            label_text = label[cls] if cls < len(label) else "Unknown"

            # Display the predicted label on the screen
            cv2.putText(img, f"{label_text} ({conf:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the main image with predictions
    cv2.imshow("Sign Language Detection", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()














'''import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO  # Import YOLOv8

# Load YOLOv8 trained model
model = YOLO("runs/runs/classify/train/weights/best.pt")  # Update path if needed

# Load webcam
cap = cv2.VideoCapture(0)

# Image processing parameters
offset = 20
imgSize = 300

# Labels (Update according to your trained model classes)
label = ["A", "B", "C"]  # Modify based on your dataset

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not access webcam.")
        break

    # Run YOLOv8 on the frame
    results = model(img)

    # Check if any detections exist
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue  # Skip if no detections

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index

            if conf > 0.3:  # Reduce confidence threshold if needed
                # Ensure bounding box is within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                imgCrop = img[y1:y2, x1:x2]  # Crop detected area

                # Ensure cropped image has valid dimensions
                if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    aspectRatio = (y2 - y1) / (x2 - x1)

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White background

                    if aspectRatio > 1:
                        k = imgSize / (y2 - y1)
                        wCal = math.ceil(k * (x2 - x1))
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / (x2 - x1)
                        hCal = math.ceil(k * (y2 - y1))
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Display cropped images
                    cv2.imshow("Cropped Image", imgCrop)
                    cv2.imshow("Processed Image", imgWhite)

                    # Assign class label
                    label_text = label[cls] if cls < len(label) else "Unknown"

                    # Draw bounding box & label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with background for better visibility
                    label_bg_x2 = x1 + len(label_text) * 17
                    cv2.rectangle(img, (x1, y1 - 30), (label_bg_x2, y1), (0, 255, 0), -1)
                    cv2.putText(img, f"{label_text} ({conf:.2f})", (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the main image with detections
    cv2.imshow("Sign Language Detection", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()'''
