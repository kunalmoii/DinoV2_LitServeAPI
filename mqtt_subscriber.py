import paho.mqtt.client as mqtt
import json
import ssl
import os
from dotenv import load_dotenv

load_dotenv()

class BayMQTTSubscriber:
    def __init__(self):
        self.client = mqtt.Client()
        self.broker_host = "ab7f2fa8.ala.dedicated.gcp.emqxcloud.com"
        self.broker_port = 1883

        # MQTT credentials from environment variables
        self.username = os.getenv("MQTT_USERNAME", "")
        self.password = os.getenv("MQTT_PASSWORD", "")


        # Set username and password if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.client.on_subscribe = self.on_subscribe

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ Connected to MQTT broker successfully!")
            # Subscribe to bay results topic
            client.subscribe("bay/predictions", qos=1)
            print("📡 Subscribed to 'bay/results' topic")
        else:
            print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            print(f"\n📨 Received message on topic '{topic}':")

            # Try to parse as JSON for better formatting
            try:
                data = json.loads(payload)
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print(payload)

        except Exception as e:
            print(f"❌ Error processing message: {e}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"⚠️ Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            print("👋 Disconnected from MQTT broker")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print(f"✅ Successfully subscribed with QoS: {granted_qos}")

    def connect(self):
        try:
            print(f"🔗 Connecting to MQTT broker: {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, 60)
            return True
        except Exception as e:
            print(f"❌ Failed to connect to MQTT broker: {e}")
            return False

    def start_listening(self):
        """Start the MQTT client loop to listen for messages"""
        try:
            print("🎧 Starting MQTT listener...")
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\n🛑 Stopping MQTT listener...")
            self.disconnect()

    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.disconnect()

    def subscribe_to_topic(self, topic, qos=1):
        """Subscribe to additional topics"""
        self.client.subscribe(topic, qos)
        print(f"📡 Subscribed to '{topic}' topic")

if __name__ == "__main__":
    subscriber = BayMQTTSubscriber()
    if subscriber.connect():
        subscriber.start_listening()
