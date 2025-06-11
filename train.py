from ultralytics import YOLO
import torch


model = YOLO("your model")


# Train the model
model.train(
    data="",  # Path to your dataset YAML
    epochs=300,  # Set the number of epochs
    batch=32,   # Batch size
    imgsz=640, # Image size
    optimizer='SGD',  # Optimizer type
    lr0=0.001,   # Learning rate
    freeze=10 #freeze 10 layers
)