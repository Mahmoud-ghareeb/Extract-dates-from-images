from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)

def get_model(path: str):
    '''
        load the tflite model
    '''
    interpreter = tf.lite.Interpreter(path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

def get_vocab(path: str):
    '''
        load the vocabulary
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
    vect_layer = tf.keras.layers.TextVectorization.from_config(data['config'])
    vect_layer.set_vocabulary(data['vocab'])

    return vect_layer

#get the model
interpreter, input_details, output_details = get_model('./Tf_lite/date_extractor.tflite')
#get the vocab
vect_layer = get_vocab('./vocab/vocab.pkl')

def inference(img):
    '''
        extract a date from an image
    '''
    img = Image.open(img)
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    image_shape = tf.shape(img)
    width_start = image_shape[1] // 4
    width_end = 3 * image_shape[1] // 4
    img = tf.slice(img, [0, width_start, 0], [-1, width_end - width_start, -1])
    img = img / 255.0

    #extractor function
    def date(img):
        img = tf.expand_dims(img, 0)
        tokens = ['G']
        inputs = tf.expand_dims(vect_layer(''.join(tokens)), 0)
        idx_to_wrd = dict(enumerate(vect_layer.get_vocabulary()))
        
        for i in range(10):
            interpreter.set_tensor(input_details[0]['index'], tf.cast(inputs, tf.float32))            
            interpreter.set_tensor(input_details[1]['index'], img)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            idx = np.argmax(predictions[0, i, :])
            wrd = idx_to_wrd[idx]
            if wrd == 'E':
                break
            tokens.append(wrd)
            inputs = tf.expand_dims(vect_layer(''.join(tokens)), 0)

        return tokens[1:]

    date = date(img)

    return ''.join(date), ''.join(date)

@app.route('/extract_date', methods=['POST'])
def extract_date():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided"})

    image = request.files['image']
    date, en_date = inference(image)

    return jsonify({"success": True, "date": date})

if __name__ == '__main__':
    app.run(debug=True)