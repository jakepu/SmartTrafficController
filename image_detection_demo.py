# start timing for performance evaluation
from datetime import datetime # time execution time
startTime = datetime.now()

import time
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


# initialize parameters
min_conf_threshold = 0.5 # 0.35 seems to work well with traffic pic
car_conf_threshold = 0.75
imH = 320
imW = 320
PATH_TO_LABELS = 'labelmap.txt'

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

# function to output objects detected in the given image and number of cars detected
# flag: define the input image and how many cars in that image
#       0 == no cars; 1 == one car; 2 == several cars; 3 == many cars
def detect(flag):
    count = 0
    if flag == 0:
        image_path = 'test_image.jpg'
    elif flag == 1:
        image_path = 'test_image1.jpg'
    elif flag == 2:
        image_path = 'test_image2.jpg'
    elif flag == 3:
        image_path = 'test_image3.jpg'
    elif flag == 4:
        image_path = 'test_image4.jpg'

    image = Image.open(image_path)
    image_cv = cv2.imread(image_path)
    input_data = np.asarray([np.asarray(image)])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    startTime_invoke = datetime.now()
    interpreter.invoke()
    print('Inference time:')
    print(datetime.now() - startTime_invoke)
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = classes + 1
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num = interpreter.get_tensor(output_details[3]['index'])[0]

    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image_cv, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            if (scores[i] > min_conf_threshold) and (object_name == 'car'):
                count += 1

            # Draw label
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image_cv, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image_cv, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1) # Draw label text

    # stop timing
    print('Total execution time:')
    print(datetime.now() - startTime)

    # All the results have been drawn on the image, now display the image
    cv2.imshow('Object detector', image_cv)
    cv2.waitKey()

    # Clean up
    #cv2.destroyAllWindows()
    return count

if __name__ == '__main__':
    num_cars = detect(3)
    print("Number of Cars Detected:")
    print(num_cars)