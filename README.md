# Extract dates from images with 100% accuracy on test set

I divided the problem into 7 steps

1. Process the data
2. Make a tf dataset
3. Build the model
4. Train the model
5. Convert the model into tf_lite and ONNX
6. Make inference
7. Build an endpoint using flask

# Process The Data

we have 2 types of data (image, text)
1. the images have a shape of (80, 500, 3)  
    the images has a big background that has no information so i cropped it and it becomes of a shape (80, 250, 3)
2. for the text the sequence length is 10 so i just converted the arabic numbers into english numbers

# Building The Dataset

first i split the data into train, test and validation  
each set has two elements (image path, encoded text)  

after that i created a tf.data.Dataset that return dictionary of 
```
{
    'enc_inputs': the image tensor of shape (None, 80, 250, 3),
    'dec_inputs': date input to decoder in the form of 'G' + the date
},
'labels' => decoder output in the form of date + 'E'
```

G and E stands for (GO) and (End) for starting and ending the generation

# Building The Model

i divided the model in to 3 parts
1. encoder
2. decoder
3. attention mechanism

1. for the  encoder part i chose to use mobilenet v3 small => (low number of parameter with high accuracy).
2. for the decoder part since it has sequence information and the sequence is short there is no need to use a transformer decoder here LSTM unit will be great and even better.
3. for the attention part i used multiheadattention layer with 1 head and key_dim of 128

# Train The Model

for training i used the following
1. adam optimizer with lr of .001
2. sparse categorical crossentropy as a loss function
3. accuracy as a metrics

# Convert The Model Into TF_LITE and ONNX

after training done i converted the model into tf_lite format the hall size of the model decreases   
<b>from 14 mb to 1.5 mb</b> and inference time <b>from 2.1 sec to 135 ms</b>

# Inference Code

Write Inference Code

# Build An Endpoint Using Flask

run the script using
```` python
python app.py
````

then send a post request with an <b>image</b> as key and an image file as a value to the endpoint
````
http://127.0.0.1:5000/extract_date
````