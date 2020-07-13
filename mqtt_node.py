import paho.mqtt.client as mqtt
import socket # socket.gethostname()
import re
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("Traffic/#")
    client.subscribe("Current_Station")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if msg.topic == "Current_Station":
        if msg.payload == machine_number:
            is_go = True
        else:
            is_go = False

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("stc1")
hostname = socket.gethostname()
machine_number = re.findall(r'\d+', hostname)[0]
is_go = False
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_start()
#client.loop_stop()
