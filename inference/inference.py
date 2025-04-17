# inference/inference.py

import os
from utils import load_model, predict_image

# === Load Class Names ===
class_names = sorted(os.listdir("TrainingData/train"))

# === Load Model ===
model_path = "Model/best_model.pth"
model, device = load_model(model_path, class_names)

# === Predict ===
if __name__ == "__main__":
    test_image_path = "TestData_Cropped/Bay_16/DMSYS_DEMO_VERTICAL_MARKET_1_1743120005.809563_image_only.jpg_Bay_16.png"
    predicted_class = predict_image(model, device, test_image_path, class_names)

    print(f"🖼️ Image: {test_image_path}")
    print(f"🔍 Predicted Class: {predicted_class}")
