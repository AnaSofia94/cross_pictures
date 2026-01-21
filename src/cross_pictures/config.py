import os

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "resources/resnet50-v2-7.onnx")
    LABELS_PATH = os.getenv("LABELS_PATH", "resources/imagenet_classes.txt")
    TOP_K = os.getenv("TOP_K", 3)
    NUM_THREADS = os.getenv("NUM_THREADS", 4)