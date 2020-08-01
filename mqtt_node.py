import paho.mqtt.client as mqtt
import socket # socket.gethostname()
import re
import car_detection_TF
client = mqtt.Client()
traffic = 0
is_go = False
payload = ""
hostname = socket.gethostname()
detector = car_detection_TF.DetectCar()
try:
    machine_number = re.search(r'\d+', hostname).group()
except: # running on device without setting up stc* hostname
    import random
    machine_number = random.randrange(3,10)
def current_station():
    global is_go, payload
    if int(payload) == machine_number:
        is_go = True
    else:
        is_go = False
def request_received():
    global client, payload, hostname, traffic
    traffic = detector.detect()
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
    print('msg.topic: ', msg.topic, 'msg.payload: ', msg.payload)
    func = switcher[msg.topic]
    func()

def init():
    global client, hostname, machine_number
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect("stc1")
    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_start()
def stop():
    global client
    client.loop_stop()
def update_traffic(num):
    global traffic
    traffic = num
if __name__ == '__main__':
    traffic = int(input('Please type a number to setup traffic for current device: '))
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("stc1")
    client.loop_forever()
