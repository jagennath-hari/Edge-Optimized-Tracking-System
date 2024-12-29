import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, image_dir: str):
        """
        Initializes the DataLoader.

        Args:
            image_dir (str): Path to the directory containing images and annotation `.txt` files.
        """
        self.image_dir = image_dir
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        )

    def load_annotations(self, annotation_path: str) -> np.ndarray:
        """
        Loads the annotation file as a numpy array.

        Args:
            annotation_path (str): Path to the `.txt` file containing annotations.

        Returns:
            np.ndarray: Array of annotations with columns [x1, y1, x2, y2, class_id, confidence].
        """
        if os.path.exists(annotation_path):
            try:
                return np.loadtxt(annotation_path, skiprows=1)  # Skip the header row
            except ValueError as e:
                print(f"Error reading {annotation_path}: {e}")
                return np.empty((0, 6))  # Return an empty array if there's an error
        else:
            return np.empty((0, 6))  # Return an empty array if no annotations exist


    def draw_boxes(self, image: np.ndarray, annotations: np.ndarray) -> np.ndarray:
        """
        Draws bounding boxes, class IDs, and confidence values on the image.

        Args:
            image (np.ndarray): The image as a numpy array.
            annotations (np.ndarray): Array of annotations with [x1, y1, x2, y2, class_id, confidence].

        Returns:
            np.ndarray: Annotated image.
        """
        for annotation in annotations:
            x1, y1, x2, y2, class_id, confidence = annotation
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw the bounding box
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for players, Blue for ball
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Add label text
            label = f"Class {int(class_id)}: {confidence:.2f}"
            cv2.putText(
                image, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return image

    def run(self):
        """
        Generator function that yields each image and its corresponding annotations.

        Yields:
            Tuple[np.ndarray, np.ndarray]: The image and its annotation array.
        """
        for image_name in self.image_files:
            # Load the image
            image_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(image_path)

            # Load the corresponding annotation
            annotation_path = os.path.splitext(image_path)[0] + '.txt'
            annotations = self.load_annotations(annotation_path)

            yield image, annotations


# Example Usage
if __name__ == "__main__":
    image_dir = "/home/hari/Downloads/SportsMOT_example/dataset/train/v_gQNyhv8y0QY_c013/img1/"
    data_loader = DataLoader(image_dir)
    data_loader.run()