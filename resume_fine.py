"""Resume the CURRENT fine-mask training run (segs_fine_s-5).

That run stopped at epoch 69/150. Ultralytics stores optimizer state, LR schedule and the
epoch counter inside last.pt, so `resume=True` picks all of that up and continues from
epoch 70 rather than restarting.

Do NOT use `optimizer=` here. Passing any optimizer override makes Ultralytics rebuild the
trainer and silently reset the schedule, which turns a resume into a fresh run. The AdamW
setting is already baked into the checkpoint.

IMPORTANT: stop `python app.py` first. Live AI inference competes for VRAM on a 4 GB card and
was what killed this run the first time.
"""
from ultralytics import YOLO

RUN = "runs/segment/runs/fine_seg/segs_fine_s-5"

if __name__ == "__main__":
    model = YOLO(f"{RUN}/weights/last.pt")
    model.train(resume=True, exist_ok=True)
    print(f"RESUMED {RUN} — when it finishes, copy weights/best.pt to crack_fine.pt")
