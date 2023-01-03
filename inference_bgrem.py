import os
import numpy as np
import argparse
import cv2 as cv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.list_physical_devices('GPU')
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image as Img

import onnx

# from onnx_tf.backend import prepare

# onnx_model = onnx.load("cloth_segm.onnx")  # load onnx model
# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph("cloth_segmentation.pb")  # export the model

model = tf.keras.models.load_model('u2net_keras.h5')

# inf = model.signatures["serving_default"]
# print(inf)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#to accept an image as an input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image",help="path to the input image")

ap.add_argument("-o","--output_image",help="path to output image")
args = vars(ap.parse_args())


def loadImage(path):
    "returns image in numpy array form"

    return cv.imread(path)

def normalize(image):

    "returns images normalize [0,1] and resized"
    # image = tf.image.per_image_standardization(image)
    image = cv.resize(image, (320, 320))
    image = image.astype('float32') / 255.
    image = np.moveaxis(image, 2, 0)
    image = np.expand_dims(image,0)
    print(image.shape,"output")
    return image

def predict(img):
    "returns the class of prediction"

    # pred = inf(tf.constant(img))

    pred = model.predict(img)
    pred = np.array(pred[0])
    # pred = list(pred.values())[0]

    predict_img = np.squeeze(pred, axis=0) #remove batch axis (1,256,256,1) => (256,256,1)
    predict_img.shape

    data = np.moveaxis(predict_img, 0, 2)
    tf.keras.utils.save_img(f'{args["output_image"]}',data)


i_image = args["input_image"]


image = loadImage(i_image)

h = image.shape[0]
w = image.shape[1]


reimage = normalize(image)
print(image.shape)

print(predict(reimage))



# BACKGROUND REMOVAL

output = load_img(f'{args["output_image"]}')

RESCALE = 255
out_img = img_to_array(output)
out_img /= RESCALE


THRESHOLD = 0.5

# refine the output
out_img[out_img > THRESHOLD] = 1
out_img[out_img <= THRESHOLD] = 0

shape = out_img.shape

a_layer_init = np.ones(shape = (shape[0],shape[1],1))
mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
a_layer = mul_layer*a_layer_init
rgba_out = np.append(out_img,a_layer,axis=2)


input = load_img(i_image)
inp_img = img_to_array(input)
inp_img= cv.resize(inp_img, (320, 320))

inp_img /= RESCALE


a_layer = np.ones(shape = (shape[0],shape[1],1))
rgba_inp = np.append(inp_img,a_layer,axis=2)

rem_back = (rgba_inp*rgba_out)
rem_back = cv.resize(rem_back,(w,h))
rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')

tf.keras.utils.save_img("bgrem.png",rem_back_scaled)



















# THRESHOLD = 0.9

# reimage[reimage > THRESHOLD] = 1
# reimage[reimage <= THRESHOLD] = 0
    
# reimage = np.squeeze(reimage, axis=0)

# shape = reimage.shape
# a_layer_init = np.ones(shape = (shape[0],shape[1],1))
# mul_layer = np.expand_dims(reimage[:,:,0],axis=2)
# a_layer = mul_layer*a_layer_init
# rgba_out = np.append(reimage,a_layer,axis=2)

# input = "pred.png"
# i_image = loadImage(input)
# i_image = normalize(i_image)
# print(i_image.shape,"shape")
# i_image = np.squeeze(i_image, axis=0)
# i_image = np.moveaxis(i_image, 0, 2)

# a_layer = np.ones(shape = (320,320,1))
# print(a_layer.shape)
# rgba_inp = np.append(i_image,a_layer,axis=2)
# print(rgba_inp.shape)

# rem_back = (rgba_inp*rgba_out)
# rem_back_scaled = rem_back*255
# data = np.moveaxis(rem_back_scaled, 0, 2)

# tf.keras.utils.save_img("final.png",data)


