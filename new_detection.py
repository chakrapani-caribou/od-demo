import cv2
import numpy as np

# Load YOLOv4 model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Read image
image_path = "Screenshot from 2025-03-19 14-35-13.png"
output_image_path = "output_image.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Cannot load image.")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# YOLO input processing
blob = cv2.dnn.blobFromImage(image, 1/255.0, (506, 506), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
print(layer_names)
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)

# Run inference
outputs = net.forward(output_layers)
print(outputs)

# Load class labels
classes = open("coco.names").read().strip().split("\n")

# Process detections
for output in outputs:
    for detection in output:
        scores = detection[5:]  # Extract class scores

        class_id = np.argmax(scores)  # Get class with highest confidence
        confidence = scores[class_id]

        if confidence > 0.5:  # Confidence threshold
            # print(detection[:4] * np.array([width, height, width, height]))
            center_x, center_y, w, h = detection[:4] * np.array([width, height, width, height])
            x1, y1 = int(center_x - w / 2), int(center_y - h / 2)
            x2, y2 = int(center_x + w / 2), int(center_y + h / 2)

            # Draw bounding box and label
            label = f"{classes[class_id]} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cropped_image = image[y1:y1+h,x1:x1+w]
            cv2.imwrite('cropped_image.jpeg',cropped_image)

# Save the processed image
cv2.imwrite(output_image_path, image)
print(f"Processed image saved as {output_image_path}")
