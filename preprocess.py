import cv2
from mtcnn import MTCNN
import os

def preprocess_faces_in_directory(input_dir, output_dir, margin=20):
    detector = MTCNN()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue

            faces = detector.detect_faces(img)

            if len(faces) == 0:
                print(f"No faces detected in {filename}")
                continue

            for i, face in enumerate(faces):
                bounding_box = face['box']
                x, y, w, h = bounding_box

                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)

                cropped_face = img[y1:y2, x1:x2]

                face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
                output_path = os.path.join(output_dir, face_filename)
                cv2.imwrite(output_path, cropped_face)
                print(f"Saved preprocessed face: {output_path}")

# Example usage
input_dir = 'datasets/oulu-npu/images/preprocessed/test/spoof' 
output_dir = 'datasets/oulu-npu/images/postprocessed/test/spoof'  

preprocess_faces_in_directory(input_dir, output_dir)
