import base64
import requests
import json

# Define the server URL
SERVER_URL = "http://0.0.0.0:8000/predict"


def predict_image(
    image_path,
    server_url=SERVER_URL,
    save_to_file=True,
    output_file="prediction_result.json",
):
    # Read and encode the image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Send the POST request
    try:
        response = requests.post(server_url, json={"image": encoded_image})
        response.raise_for_status()
        prediction_result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

    # Print the result
    print("✅ Prediction:", prediction_result)

    # Optionally save to JSON
    if save_to_file:
        with open(output_file, "w") as json_file:
            json.dump(prediction_result, json_file, indent=4)
        print(f"✅ Prediction saved to '{output_file}'")

    return prediction_result


if __name__ == "__main__":
    image_path = "test.jpg"  # Path to your test image
    server_url = SERVER_URL  # Use the predefined server URL

    # Call the prediction function
    predict_image(image_path, server_url)
