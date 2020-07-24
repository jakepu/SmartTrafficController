import paho.mqtt.client as mqtt
import socket # socket.gethostname()
import re
client = mqtt.Client()
traffic = 0
is_go = False
payload = ""
hostname = socket.gethostname()
machine_number = re.search(r'\d+', hostname).group()
def current_station():
    global is_go, payload
    if int(payload) == machine_number:
        is_go = True
    else:
        is_go = False
def request_received():
    global payload, hostname
    payload = hostname + ": Traffic - " + str(traffic)
    client.publish("Traffic", payload = payload)

# table of function to call when processing incoming message
switcher = {"Current_Station": current_station, "Request": request_received}

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("Current_Station")
    client.subscribe("Request")
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global payload, switcher
    payload = msg.payload
    switcher[msg.topic]

def init():
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect("stc1")
    hostname = socket.gethostname()
    machine_number = re.findall(r'\d+', hostname)[0]
    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_start()
    #client.loop_stop()
def update_traffic(num):
    global traffic
    traffic = num
if __name__ == '__main__':
    traffic = int(input('Please type a number to setup traffic for current device'))
    init()
