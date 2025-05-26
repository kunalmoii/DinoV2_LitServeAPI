import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import json
import random
import time

BROKER_HOST = "l7dcf52a.ala.eu-central-1.emqxsl.com"
BROKER_PORT = 8883
TOPIC = "bay/predictions"
USERNAME = "dms_mqtt"
PASSWORD = "*kk9t@RI!9RaTN"
USE_TLS = True
CA_CERTS = "emqxsl-ca.crt"


class BayMQTTReceiver:
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
        self.message_received = False
        random.seed(1)  # Different seed from publisher
        client_id = f"bay-receiver-{random.randint(0, 10000)}"
        self.client = mqtt.Client(CallbackAPIVersion.VERSION1, client_id)

        if username and password:
            self.client.username_pw_set(username, password)
        if use_tls and ca_certs:
            self.client.tls_set(ca_certs=ca_certs)

        # Set up callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Receiver connected to MQTT Broker!")
            print(f"Subscribing to topic: {self.topic}")
            client.subscribe(self.topic)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            print(f"\n--- Received message on topic: {msg.topic} ---")
            payload = msg.payload.decode()
            print(f"Raw payload: {payload}")

            # Try to parse as JSON
            bay_results = json.loads(payload)
            print("Parsed bay results:")
            for result in bay_results:
                print(f"  Bay: {result.get('bay', 'N/A')}")
                print(f"  Prediction: {result.get('prediction', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 'N/A')}")
                print(f"  Image Path: {result.get('image_path', 'N/A')}")
                print("  ---")
            self.message_received = True
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {payload}")
            self.message_received = True
        except Exception as e:
            print(f"Error processing message: {e}")
            self.message_received = True

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from MQTT Broker")

    def connect_and_listen(self):
        """Connect to broker and start listening for messages."""
        try:
            self.client.connect(BROKER_HOST, BROKER_PORT)
            self.client.loop_start()
            print("MQTT Receiver started. Waiting for one message...")

            # Wait for one message
            while not self.message_received:
                time.sleep(0.1)

            print("Message received. Exiting...")

        except KeyboardInterrupt:
            print("\nStopping receiver...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.disconnect()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("Receiver disconnected")


if __name__ == "__main__":
    receiver = BayMQTTReceiver()
    receiver.connect_and_listen()
