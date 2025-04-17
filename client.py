import base64
import requests

# Read and encode the image
with open("test.jpg", "rb") as f:
    image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

# Send the POST request
# response = requests.post("http://localhost:8000/", json={"image": encoded_image})
response = requests.post("http://localhost:8000/predict", json={"image": encoded_image})


# Print the result
print("✅ Prediction:", response.json())
