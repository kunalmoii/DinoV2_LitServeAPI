import paho.mqtt.client as mqtt
import json
import random

BROKER_HOST = "l7dcf52a.ala.eu-central-1.emqxsl.com"
BROKER_PORT = 8883
TOPIC = "bay/predictions"
USERNAME = "dms_mqtt"
PASSWORD = "*kk9t@RI!9RaTN"
USE_TLS = True
CA_CERTS = "emqxsl-ca.crt"  # <-- Update this path to your CA cert file


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
        client_id = f"bay-mqtt-{random.randint(0, 10000)}"
        self.client = mqtt.Client(client_id)
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
        bay_results: list of dicts, e.g.
        [
            {"bay": "A1", "prediction": "Empty", "confidence": 0.98},
            ...
        ]
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
