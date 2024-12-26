import argparse
from ultralytics import YOLO

def main(data_path):
    # Load a model
    model = YOLO("yolo11s.yaml")  # build a new model from YAML
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=data_path, epochs=300, imgsz=736, device=[0], batch=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with custom dataset path")
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Path to the data.yaml file for YOLO training"
    )
    args = parser.parse_args()

    # Call the main function with the dataset path
    main(args.data)
