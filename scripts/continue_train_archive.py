from ultralytics import YOLO
import os
import shutil

if __name__ == '__main__':
    for f in ['crack_dataset/train/labels.cache', 'crack_dataset/test/labels.cache']:
        if os.path.exists(f):
            os.remove(f)

    model = YOLO('crack_trained.pt')
    results = model.train(
        data='crack_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        exist_ok=True,
        name='seg_train_v2'
    )
    print("Segmentation fine-tune complete!")
    best = f"{results.save_dir}/weights/best.pt"
    print(f"Best model: {best}")
    shutil.copy2(best, 'crack_trained.pt')
    print("Copied best.pt -> crack_trained.pt")
