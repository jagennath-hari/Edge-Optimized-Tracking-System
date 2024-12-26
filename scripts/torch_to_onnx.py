import argparse
from ultralytics import YOLO

def main(pt_path):
    # Load a model
    model = YOLO(pt_path)  # Load a custom trained model

    # Export the model
    model.export(format="onnx", device=[0], imgsz=736, half=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument(
        "--pt", 
        type=str, 
        required=True, 
        help="Path to the .pt file of the YOLO model"
    )
    args = parser.parse_args()

    # Call the main function with the .pt file path
    main(args.pt)
