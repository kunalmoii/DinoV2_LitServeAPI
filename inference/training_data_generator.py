import cv2
import numpy as np
from PIL import Image
import json
import os
from settings import __APP_SETTINGS__


def crop_bay_regions_from_image(
    image_path: str,
    bay_coordinates_path: str,
    output_dir: str,
    input_size=(1920, 1080),
    original_size=(1920, 1080),
):
    # Load JSON with bay coordinates
    bay_coordinates = json.load(open(bay_coordinates_path, "r"))

    # Scaling factors
    x_scale = input_size[0] / original_size[0]
    y_scale = input_size[1] / original_size[1]

    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.basename(image_path).replace(".png", "")
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    for bay_name, coords in bay_coordinates.items():
        if not coords:
            print(f"⚠️ No coordinates found for {bay_name}, skipping.")
            continue

        # Ensure polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        # Scale coordinates
        scaled_coords = [[int(x * x_scale), int(y * y_scale)] for x, y in coords]
        polygon = np.array(scaled_coords, dtype=np.int32)

        # Create mask and apply it
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Get bounding box and crop
        x, y, w, h = cv2.boundingRect(polygon)
        x_end, y_end = min(x + w, width), min(y + h, height)

        if x >= width or y >= height or x_end <= x or y_end <= y:
            print(f"⚠️ Skipping invalid crop for {bay_name}")
            continue

        cropped = masked[y:y_end, x:x_end]

        # Save cropped image
        bay_dir = os.path.join(output_dir, bay_name)
        os.makedirs(bay_dir, exist_ok=True)

        out_path = os.path.join(bay_dir, f"{image_name}_{bay_name}.png")
        cv2.imwrite(out_path, cropped)
        print("✅ Saved:", out_path)


# === Example usage ===
if __name__ == "__main__":
    crop_bay_regions_from_image(
        image_path="TestData/DMSYS_DEMO_VERTICAL_MARKET_1_1743120005.809563_image_only.jpg",  # Replace with your actual image
        bay_coordinates_path=__APP_SETTINGS__.BAY_COORDINATES_PATH,
        output_dir="TestData_Cropped",
        input_size=(1920, 1080),
        original_size=(1920, 1080),
    )
