import tensorflow as tf
import os
converter = tf.lite.TFLiteConverter.from_saved_model(os.getcwd())
tflite_model = converter.convert()
with tf.io.gfile.GFile('model.tflite','wb') as f:
    f.write(tflite_model)
print('TFLite model conversion succesful. Saved to model.tflite')