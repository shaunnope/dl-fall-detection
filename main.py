from ultralytics import YOLO

# model= YOLO("yolov8n.yaml")
model= YOLO("best.pt")

if __name__ == "__main__":
  results= model.train(data="config.yaml", epochs=50)