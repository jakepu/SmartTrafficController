# Paho MQTT server

import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
#from threading import Thread
from time import sleep
import re
from socket import gethostname
import car_detection_TF
hostname = gethostname() # should be stc1
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
SERVICE_VEHICLE_NUM = 0
checkpoint_counter = 0
current_station_name = hostname
status_update_online = False
detector = car_detection_TF.DetectCar()
def current_station():
    global is_go, machine_number
    if payload == machine_number:
        is_go = True
    else:
        is_go = False
def check_intersection():
    global stations_dict, payload 
    hostname, _, traffic = payload.partition(': Traffic - ')
    stations_dict[hostname] = int(traffic)
    online_stations.append(hostname)
    #wait_dict[hostname] = 1+wait_dict[hostname] if hostname in wait_dict else 1
    if hostname not in wait_dict:
        wait_dict[hostname] = 0
def sum_checkpoint_traffic():
    global payload, checkpoint_counter
    checkpoint, _, traffic = payload.partition(': Traffic - ')
    checkpoint_counter += int(traffic)
# table of function to call when processing incoming message
switcher = {"Traffic": check_intersection, "Current_Station": current_station, "Checkpoint": sum_checkpoint_traffic}

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("Traffic") # nodes at the intersections
    client.subscribe("Current_Station") # the only station that is open
    client.subscribe("Checkpoint") # cameras covering the area
    #client.subscribe("Request")
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global payload, switcher
    payload = msg.payload.decode()
    print('msg.topic: ', msg.topic, 'msg.payload: ', msg.payload)
    func = switcher.get(msg.topic)
    func()

def request():
    global client, stations_dict, online_stations, hostname, checkpoint_counter
    stations_dict.clear()
    online_stations = [hostname]
    checkpoint_counter = 0
    client.publish("Request", 1)
def status_update():
    global hostname, traffic, stations_dict
    while status_update_online:
        request()
        sleep(5) # wait for other stations to feedback their traffic
        traffic = detector.detect()
        stations_dict[hostname] = int(traffic) # input the server's own traffic
        choose_current_station()
        sleep(10)

def choose_current_station():
    global stations_dict, online_stations, wait_dict, traffic, hostname, checkpoint_counter, current_station_name
    if checkpoint_counter <= SERVICE_VEHICLE_NUM: # if there are traffic in the area, no change in station
        current_station_name = hostname
        longest_wait = 0
        for station in online_stations:
            if wait_dict[station] >= 3:
                current_station_name = station
                longest_wait = wait_dict[station]
            wait_dict[station] += 1
        if longest_wait > 0:
            wait_dict[station] = 0
            print('current station: ', current_station_name)
            wait_dict[current_station_name] = 0
            return current_station_name
        rank = sorted(stations_dict.items(), key= lambda item: item[1], reverse=True)
        for i in range(len(rank)):
            if rank[i][0] in online_stations and rank[i][1] >= traffic:
                current_station_name = rank[i][0]
                break
        wait_dict[current_station_name] = 0
    print('current station: ', current_station_name)
    return current_station_name
def init():
    global client, hostname, wait_dict
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname, 1883, 60)
    # initialize wait_dict and put server in it
    wait_dict[hostname] = 0
    # start a thread to handle network traffic
    client.loop_start()
    status_update_thread.start()
def stop():
    global client
    client.loop_stop()
    status_update_online = False
def update_traffic(num):
    global traffic
    traffic = num
def update_service_vehicle_num(num):
    global SERVICE_VEHICLE_NUM
    SERVICE_VEHICLE_NUM = num
if __name__ == '__main__':
    from threading import Thread
    traffic = int(input('Please type a number to setup traffic for current device: '))
    status_update_thread = Thread(target = status_update)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("stc1", 1883, 60)
    # initialize wait_dict and put server in it
    wait_dict[hostname] = 0
    client.loop_start()
    status_update_online = False
    status_update_thread.start()
    
    