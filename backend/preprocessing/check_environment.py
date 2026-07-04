import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("RAASTA Environment Check")
print("=" * 50)

print(f"PyTorch Version : {torch.__version__}")
print(f"OpenCV Version  : {cv2.__version__}")
print(f"NumPy Version   : {np.__version__}")

print("\nCUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Using CPU (We'll train on Google Colab later)")

print("\nEverything is working successfully!")