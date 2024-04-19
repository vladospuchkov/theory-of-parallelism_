import cv2
import argparse
from ultralytics import YOLO
import torch 



# madel = YOLO('yolov8n-pose.yaml')
model = YOLO('yolov8n-pose.pt')


res = model("input.mp4", save=True, device="cpu")

# model.MODE(ARGS)