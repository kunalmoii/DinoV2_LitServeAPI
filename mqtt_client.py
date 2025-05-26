import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import json
import random

BROKER_HOST = "l7dcf52a.ala.eu-central-1.emqxsl.com"
BROKER_PORT = 8883
TOPIC = "bay/predictions"
USERNAME = "dms_mqtt"
PASSWORD = "*kk9t@RI!9RaTN"
USE_TLS = True
CA_CERTS = "emqxsl-ca.crt"


class BayMQTTPublisher:
    def __init__(
        self,
        broker_host=BROKER_HOST,
        broker_port=BROKER_PORT,
        topic=TOPIC,
        username=USERNAME,
        password=PASSWORD,
        use_tls=USE_TLS,
        ca_certs=CA_CERTS,
    ):
        self.topic = topic
        random.seed(0)
        client_id = f"bay-mqtt-{random.randint(0, 10000)}"
        self.client = mqtt.Client(CallbackAPIVersion.VERSION1, client_id)
        if username and password:
            self.client.username_pw_set(username, password)
        if use_tls and ca_certs:
            self.client.tls_set(ca_certs=ca_certs)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print(f"Failed to connect, return code {rc}")

        self.client.on_connect = on_connect
        self.client.connect(broker_host, broker_port)
        self.client.loop_start()

    def _process_prediction(self, prediction, confidence):
        """Helper method to process prediction with consistency."""
        # Simplify the prediction logic
        if (prediction == "Object" and confidence < 0.99) or (
            prediction == "Occlusion" and confidence < 0.5
        ):
            prediction = "Empty"
            confidence = 1.0
        return {"prediction": prediction, "confidence": confidence}

    def publish_bay_results(self, bay_results):
        """
        Publishes the processed bay detection results to the MQTT topic.

        The input `bay_results` is a list of dictionaries, where each dictionary
        represents a bay and its initial prediction and confidence.
        Example input `bay_results`:
        [
            {"bay": "A1", "prediction": "Empty", "confidence": 0.98, "image_path": "/path/to/image_A1.jpg"},
            {"bay": "B2", "prediction": "Object", "confidence": 0.995, "image_path": "/path/to/image_B2.jpg"}
        ]

        This method processes each result using `_process_prediction` to potentially
        adjust the prediction and confidence based on predefined thresholds.
        For example, an "Object" with confidence < 0.99 might be changed to "Empty"
        with confidence 1.0.

        The final output published to the MQTT topic is a JSON string representing
        a list of these processed results. Each item in the list will be a dictionary
        containing the bay identifier, the final prediction, the final confidence,
        and the original image path.

        Example JSON payload published to the topic:
        '[
            {"bay": "A1", "prediction": "Empty", "confidence": 0.98, "image_path": "/path/to/image_A1.jpg"},
            {"bay": "B2", "prediction": "Object", "confidence": 0.995, "image_path": "/path/to/image_B2.jpg"}
        ]'
        If a prediction was changed by `_process_prediction`, it would look like:
        '[
            {"bay": "C3", "prediction": "Empty", "confidence": 1.0, "image_path": "/path/to/image_C3.jpg"}
        ]'
        (Assuming C3 was initially "Object" with confidence 0.9)

        Args:
            bay_results (list): A list of dictionaries, where each dictionary
                                contains 'bay' (str), 'prediction' (str),
                                'confidence' (float), and 'image_path' (str).
        """
        cleaned_results = []
        for result in bay_results:
            processed = self._process_prediction(
                result["prediction"], result["confidence"]
            )
            cleaned_result = result.copy()
            cleaned_result.update(processed)
            cleaned_results.append(cleaned_result)
        payload = json.dumps(cleaned_results)
        result = self.client.publish(self.topic, payload)
        status = result[0] if isinstance(result, tuple) else result.rc
        if status == 0:
            print(f"Sent results to topic {self.topic}")
        else:
            print(f"Failed to send message to topic {self.topic}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
