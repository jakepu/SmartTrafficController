# Paho MQTT server

import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

NUM_STATIONS = 2
count = NUM_STATIONS
dict_stations = {}

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("Traffic/#")
    client.subscribe("Current_Station")
    client.subscribe("Station_Information")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global count 
    dict_stations[msg.topic] = msg.payload
    count = count - 1
    if count == 0:
        sorted_dict = sorted(dict_stations.items(), key=lambda x: x[1], reverse = True)
        client.publish("Current_Station", sorted_dict[0][0], hostname="stc1")
        count = NUM_STATIONS
        dict_stations.clear()


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("stc1", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_start()

# Unresovled issues:
# how to read information from this device and add it to the dict
# how to request other node devices for information
