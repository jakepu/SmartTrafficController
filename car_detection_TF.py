import picamera
import time
import numpy as np
import tensorflow as tf
# initialize PiCamera 
camera = picamera.PiCamera()
camera.resolution = (320,320)
camera.framerate = 32
rawCapture = picamera.array.PiRGBArray(camera)
time.sleep(2)
#

