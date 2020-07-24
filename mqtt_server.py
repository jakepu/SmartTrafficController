# Paho MQTT server

import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
from multiprocessing import Process
#from threading import Thread
from time import sleep
import re
import socket # socket.gethostname()
hostname = socket.gethostname()
machine_number = re.search(r'\d+', hostname).group()
NUM_STATIONS = 2
count = NUM_STATIONS
stations_dict = {}
is_go = False
payload = ""
traffic = 0
wait_dict = {}
online_stations = []
client = mqtt.Client()
def current_station():
    global is_go, machine_number
    if payload == machine_number:
        is_go = True
    else:
        is_go = False
def read_message():
    global stations_dict, payload 
    hostname, _, traffic = payload.partition(': Traffic - ')
    stations_dict[hostname] = int(traffic)
    online_stations.append(hostname)
    #wait_dict[hostname] = 1+wait_dict[hostname] if hostname in wait_dict else 1
    if hostname not in wait_dict:
        wait_dict[hostname] = 0

# table of function to call when processing incoming message
switcher = {"Traffic": read_message, "Current_Station": current_station}

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("Traffic")
    client.subscribe("Current_Station")
    #client.subscribe("Request")
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global payload, switcher
    payload = msg.payload.decode()
    print('msg.topic: ', msg.topic, 'msg.payload: ', msg.payload)
    func = switcher.get(msg.topic)
    func()

def request():
    global client, stations_dict, online_stations
    stations_dict.clear()
    online_stations.clear()
    client.publish("Request", 1)
def status_update():
    global hostname, traffic, stations_dict
    while True:
        request()
        sleep(5) # wait for other stations to feedback their traffic
        stations_dict[hostname] = int(traffic) # input the server's own traffic
        choose_current_station()
        sleep(30)
status_update_process = Process(target = status_update)

def choose_current_station():
    global stations_dict, online_stations, wait_dict, traffic
    current_station = 'stc1'
    longest_wait = 0
    for station in online_stations:
        if wait_dict[station] >= 3:
            current_station = station
            longest_wait = wait_dict[station]
    if longest_wait > 0:
        wait_dict[station] = 0
        print('current station: ', current_station)
        return current_station
    rank = sorted(stations_dict.items(), key= lambda item: item[1], reverse=True)
    for i in range(len(rank)):
        if rank[i][0] in online_stations and rank[i][1] > traffic:
            current_station = rank[i][0]
            break
    print('current station: ', current_station)
    return current_station
def init():
    global client, status_update_process
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect("stc1", 1883, 60)

    # start a thread to handle network traffic
    client.loop_start()
    status_update_process.start()
def stop():
    global client, status_update_process
    client.loop_stop()
    status_update_process.terminate()
def update_traffic(num):
    global traffic
    traffic = num
def test_print():
    print('prints every 5 secs')
if __name__ == '__main__':
    from threading import Thread
    traffic = int(input('Please type a number to setup traffic for current device: '))
    status_update_thread = Thread(target = status_update)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("stc1", 1883, 60)
    client.loop_start()
    status_update_thread.start()
    