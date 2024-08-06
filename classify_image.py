import tensorflow as tf
import numpy as np
import cv2
import os

# Load the labels
labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load the pre-trained model
model = tf.saved_model.load('models/ssd_mobilenet_v2_coco_2018_03_29/saved_model')
detect_fn = model.signatures['serving_default']

# Function to load an image
def load_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

# Function to perform object detection
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    return detections

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, detections):
    h, w, _ = image.shape
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detection_boxes = detections['detection_boxes']
    detection_classes = detections['detection_classes'].astype(np.int64)
    detection_scores = detections['detection_scores']

    for i in range(num_detections):
        score = detection_scores[i]
        if score > 0.5:
            bbox = detection_boxes[i]
            class_id = detection_classes[i]
            label = labels[class_id - 1]  # Adjust index for 0-based labels list

            print(f"Detected class ID: {class_id}, Label: {label}, Score: {score}")  # Debug info

            y_min, x_min, y_max, x_max = bbox
            (startX, startY, endX, endY) = (int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h))

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Main function to classify image
def classify_image(image_path):
    image, image_rgb = load_image(image_path)
    detections = detect_objects(image_rgb)
    output_image = draw_boxes(image, detections)

    output_path = os.path.splitext(image_path)[0] + '_output.png'
    cv2.imwrite(output_path, output_image)
    print(f"Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path of the image: ")
    classify_image(image_path)
