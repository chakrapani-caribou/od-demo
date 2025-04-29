import cv2
import numpy as np
import time
import argparse

from numpy.ma.core import minimum

input_file_path = ["video_input.mp4","video_input_Iframes.mp4","video_input_Pframes.mp4","video_input_Bframes.mp4",
                   "video_input1.mp4","video_input1_Iframes.mp4","video_input1_Pframes.mp4","video_input1_Bframes.mp4",
                   "video_input2.mp4","video_input2_Iframes.mp4","video_input2_Pframes.mp4","video_input2_Bframes.mp4",
                   "video_input3.mp4","video_input3_Iframes.mp4","video_input3_Pframes.mp4","video_input3_Bframes.mp4"]
output_file_path = ["video_output.mp4","video_output_Iframes.mp4","video_output_Pframes.mp4","video_output_Bframes.mp4",
                   "video_output1.mp4","video_output1_Iframes.mp4","video_output1_Pframes.mp4","video_output1_Bframes.mp4",
                   "video_output2.mp4","video_output2_Iframes.mp4","video_output2_Pframes.mp4","video_output2_Bframes.mp4",
                   "video_output3.mp4","video_output3_Iframes.mp4","video_output3_Pframes.mp4","video_output3_Bframes.mp4"]

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Object Detection with Variable FPS")
parser.add_argument("--fps", type=int, help="Frames per second for processing (default: original FPS)")
args = parser.parse_args()

for video_index in range(len(output_file_path)):
    print(f"************** start of file name {input_file_path[video_index]} *******************")
    # Open video file
    video_path = input_file_path[video_index]
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    # Get original video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set FPS to default original FPS if not provided
    target_fps = args.fps if args.fps else original_fps

    # Define codec and create VideoWriter
    out = cv2.VideoWriter(f'yolov4_tiny_{output_file_path[video_index]}', cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (frame_width, frame_height))

    # Read class labels
    classes = open("coco.names").read().strip().split("\n")

    # Load YOLO model
    model = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    # Calculate frame skip interval
    frame_skip = max(1, original_fps // target_fps)

    # Start overall timer
    total_start_time = time.time()

    frame_count = 0
    processed_count = 0
    total_confidence = 0
    total_objects_detected = 0
    min_confidence_level = float('inf')
    max_confidence_level = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frame is read

        frame_count += 1

        # Process only selected frames based on frame_skip interval
        # if frame_count % frame_skip != 0:
        #     continue  # Skip this frame

        processed_count += 1
        start_time = time.time()

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)
        detections = model.forward(output_layers)

        # Store detected objects
        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.80:
                    center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        total_confidence += sum(confidences)
        total_objects_detected += len(confidences)
        temp_min_confidence = float('inf')
        temp_max_confidence = 0

        for confidence in confidences:
            temp_min_confidence = min(temp_min_confidence,confidence)
            temp_max_confidence = max(temp_max_confidence,confidence)
        min_confidence_level = min(temp_min_confidence,min_confidence_level)
        max_confidence_level = max(temp_max_confidence, max_confidence_level)

        # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed frame
        out.write(frame)

        # Calculate time taken for this frame
        frame_time = time.time() - start_time
        # print(f"Processed Frame {processed_count} at {target_fps} FPS in {frame_time:.4f} seconds")

    # Calculate total processing time
    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processed {processed_count} frames from {frame_count} total frames")
    if processed_count > 0:
        print(f"Average time per processed frame: {total_time/processed_count:.4f} seconds")
    print(f"Total confidence is: {total_confidence}")
    print(f"total no of objects: {total_objects_detected}")
    if total_objects_detected >0:
        print(f"average confidence: {total_confidence/total_objects_detected}")
    print(f"minimum confidence level: {min_confidence_level}")
    print(f"maxmimum confidence level: {max_confidence_level}")
    # Release resources
    cap.release()
    out.release()
    print(f"************** end of file name yolov4_tiny_{output_file_path[video_index]}' *******************")
    print()
