import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image as PILImage
from io import BytesIO
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.models import *
import numpy as np
import pickle as pkl
from PIL import Image,ImageChops, ImageEnhance
import cv2
import base64


# Your existing functions and imports go here...

#Function for ela 
def convert_to_ela_image(image, quality):
    temp_filename = 'temp_file.jpg'
    # ela_filename = 'temp_ela_file.png'
    
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

#Function for filters 
import numpy as np
q = [4.0, 12.0, 2.0]
filter1 = [[0, 0, 0, 0, 0],
           [0, -1, 2, -1, 0],
           [0, 2, -4, 2, 0],
           [0, -1, 2, -1, 0],
           [0, 0, 0, 0, 0]]
filter2 = [[-1, 2, -2, 2, -1],
           [2, -6, 8, -6, 2],
           [-2, 8, -12, 8, -2],
           [2, -6, 8, -6, 2],
           [-1, 2, -2, 2, -1]]
filter3 = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]


filter1 = np.asarray(filter1, dtype=float) / q[0]
filter2 = np.asarray(filter2, dtype=float) / q[1]
filter3 = np.asarray(filter3, dtype=float) / q[2]
    
filters = filter1+filter2+filter3



image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0


    
#Load model 
json_file = open('v1model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("v1model.h5")


#Load model for phase 2 
# load json and create model
json_file2 = open('dunetm.json', 'r')
loaded_model_json = json_file2.read()
json_file2.close()
#load weights 
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("dunet.h5")

def predict(image,model) :
    ela_img=prepare_image(image)
    ela_img=ela_img.reshape(1,128,128,3)
    prediction=model.predict(ela_img)
    
    return ela_img,prediction


def predict_region(img, load_model):
    img = np.array(img)
    temp_img_arr = cv2.resize(img, (512, 512))
    temp_preprocess_img = cv2.filter2D(temp_img_arr, -1, filters)
    temp_preprocess_img = cv2.resize(temp_preprocess_img, (512, 512))
    temp_img_arr = temp_img_arr.reshape(1, 512, 512, 3)
    temp_preprocess_img = temp_preprocess_img.reshape(1, 512, 512, 3)
    model_temp = load_model.predict([temp_img_arr, temp_preprocess_img])
    model_temp = model_temp[0].reshape(512, 512)
    for i in range(model_temp.shape[0]):
        for j in range(model_temp.shape[1]):
            if model_temp[i][j] > 0.75:
                model_temp[i][j] = 1.0
            else:
                model_temp[i][j] = 0.0
    return model_temp



app = FastAPI()
    

@app.post("/detect-forgery")
async def detect_forgery(file: UploadFile = File(...)):
    contents = await file.read()  # Read the contents of the uploaded file as bytes
    img = Image.open(io.BytesIO(contents))  # Open the image from the bytes content
    # Perform operations on the image (e.g., img_pil.thumbnail(), img_pil.save(), etc.)
    link_url = None
    ela_img, pred = predict(img, model)
    pred = pred[0]
    if pred >= 0.5:
        classification = "Real"
        forgery_map = predict_region(img, loaded_model)
    else:
        classification = "Fake"
        forgery_map = predict_region(img, loaded_model)
        forgery_map = predict_region(img, loaded_model)
        forgery_map = forgery_map.reshape((512,512))
        region_img = Image.fromarray((forgery_map * 255).astype(np.uint8))
        buffer = io.BytesIO()
        region_img.save(buffer, format='PNG')
        encoded_region = base64.b64encode(buffer.getvalue()).decode('utf-8')
        link_url = f"data:image/png;base64,{encoded_region}"
        

    # reshaped_img = ela_img
    ela_img = ela_img.reshape((128, 128, 3))
    ela_img_pil = Image.fromarray((ela_img * 255).astype(np.uint8))

# Convert the PIL Image to a BytesIO object and encode it as base64
    buffer = io.BytesIO()
    ela_img_pil.save(buffer, format='PNG')  # Save as JPEG or other desired format
    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Create a data URL
    data_url = f"data:image/png;base64,{encoded_img}"


    response = {
        "error_level_analysis": data_url,
        "probability_real": float(pred[0]),
        "probability_fake": float(1 - pred[0]),
        "classification": classification,
        "forgery_map": link_url
    }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
