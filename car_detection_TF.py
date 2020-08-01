# start timing for performance evaluation
from datetime import datetime # time execution time
import picamera
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

MIN_CONF_THRESHHOLD = 0.5 # 0.35 seems to work well with traffic pic
class DetectCar:
    def __init__(self):
        # initialize PiCamera 
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640,480)
        time.sleep(2)
        #self.camera.capture(rawCapture, 'rgb',resize=(320,320))
        # initialize parameters
        self.min_conf_threshold = 0.5 # 0.35 seems to work well with traffic pic
        self.imH = 320
        self.imW = 320
        self.PATH_TO_LABELS = 'labelmap.txt'

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        self.input_shape = self.input_details[0]['shape']

    def detect(self, pic = 0): 
        car_count = 0

        if pic == 0:
            self.camera.capture('output.jpg')
            image_path = 'output.jpg'
        else:
            try:
                int(pic)
                image_path = 'test_image' + str(pic) + '.jpg'
            except:
                image_path = pic

        image = Image.open(image_path)
        image = image.resize((self.imW, self.imH))
        image_cv = cv2.imread(image_path) # keep in mind that cv2 open in BGR format
        input_data = np.asarray([np.asarray(image)])
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        startTime_invoke = datetime.now()
        self.interpreter.invoke()
        print('Inference time:')
        print(datetime.now() - startTime_invoke)
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = classes + 1
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        

        for i in range(len(scores)):
            if (scores[i] > self.min_conf_threshold):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * self.imH)))
                xmin = int(max(1,(boxes[i][1] * self.imW)))
                ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                
                cv2.rectangle(image_cv, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                if (scores[i] > self.min_conf_threshold) and ((object_name == 'car') or (object_name == 'motorcycle') or 
                (object_name == 'bus') or (object_name == 'truck')): 
                    car_count += 1

                # Draw label
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image_cv, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image_cv, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1) # Draw label text

        # All the results have been drawn on the image, now display the image
        cv2.imshow('Object detector', image_cv)
        cv2.waitKey(500)

        # Clean up
        #cv2.destroyAllWindows()
            
        return car_count
if __name__ == '__main__':
    detector = DetectCar()
    print('Number of traffic:', detector.detect(1))