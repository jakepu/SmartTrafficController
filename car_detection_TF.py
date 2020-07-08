import picamera
from picamera.array import PiRGBArray 
import time
import numpy as np
import tensorflow as tf

# initialize PiCamera 
camera = picamera.PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera,size=(320,320))
camera.capture(rawCapture, 'rgb')
time.sleep(2)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array([rawCapture.array])
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
classes = interpreter.get_tensor(output_details[1]['index'])[0]
score = interpreter.get_tensor(output_details[2]['index'])[0]
num = interpreter.get_tensor(output_details[3]['index'])[0]
print("class:")
print(classes)
print("score:")
print(score)
print("num:")
print(num)


