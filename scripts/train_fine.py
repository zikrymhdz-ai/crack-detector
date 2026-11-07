"""Train fine-mask crack segmenter on fine_dataset."""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8s-seg.pt")
    model.train(
        data="fine_dataset/data.yaml",
        project="runs/fine_seg",
        name="segs_fine_s",
        imgsz=640,
        batch=8,
        workers=0,
        epochs=150,
        patience=30,
        device=0,
        seed=20,
        close_mosaic=10,
        pretrained=True,
        optimizer="AdamW",
    )


if __name__ == "__main__":
    main()