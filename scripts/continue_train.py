from ultralytics import YOLO
import os

if __name__ == '__main__':
    for f in ['crack_dataset/train/labels.cache', 'crack_dataset/test/labels.cache']:
        if os.path.exists(f):
            os.remove(f)

    model = YOLO('yolov8n-seg.pt')
    results = model.train(
        data='crack_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        exist_ok=True,
        name='seg_train'
    )
    print("Segmentation training complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")