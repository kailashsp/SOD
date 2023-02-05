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
import skimage
import base64
from PIL import Image
import io



model = tf.keras.models.load_model('u2net_keras.h5')

# inf = model.signatures["serving_default"]
# print(inf)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def loadImage(base64str):
    "returns image in numpy array form"
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    image = Image.open(bytesObj) 
    image = cv.cvtColor(np.array(image),cv.IMREAD_COLOR)
    # image = cv.imread(path)
    print(f"this is the image shape{image.shape}")
    h = image.shape[0]
    w = image.shape[1]
    
    return image, h, w

    


def normalize(image):

    "returns images normalize [0,1] and resized"
    # image = tf.image.per_image_standardization(image)

    image = cv.resize(image, (320, 320))
    # image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    print(image.shape)
    image = image.astype('float32') / 255.
    image = np.moveaxis(image, 2, 0)
    image = np.expand_dims(image,0)
    print(image.shape,"output")
    return image

def sharpen(img):
    " sharpens the images for better segmentation"
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv.filter2D(src=img,ddepth= -1, kernel =kernel)
    return image_sharp



def predict(img, height, width):
    "returns the class of prediction"

    pred = model.predict(img)
    pred = np.array(pred[0])
    predict_img = np.squeeze(pred, axis=0) #remove batch axis (1,256,256,1) => (256,256,1)
    predict_img.shape
    data = np.moveaxis(predict_img, 0, 2)
    seg = np.expand_dims(cv.resize(data, (width, height)),axis=2)
    # seg = cv.cvtColor(seg, cv.IMREAD_COLOR,dstCn=3)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # im_pil = Image.fromarray(seg)
    print("seg shape",seg.shape)
    tf.keras.utils.save_img(f"seg.png",seg)
    seg = load_img("seg.png")

    return seg

# BACKGROUND REMOVAL
def back_rem(out_img, inp_img):
    


    RESCALE = 255
    out_img = img_to_array(out_img)
    print(len(out_img))
    out_img = out_img/RESCALE


    THRESHOLD = 0.9
    # refine the output
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    shape = out_img.shape

    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)

    # tf.keras.utils.save_img("rgbaout.png",rgba_out)

    inp_img = inp_img/RESCALE


    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)

    # tf.keras.utils.save_img("rgbinp.png",rgba_inp)

    rem_back = (rgba_inp*rgba_out)
 
    # rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')
    rem_back_scaled =(rem_back*RESCALE).astype('uint8')
    rem_back_scaled= cv.cvtColor(rem_back_scaled, cv.COLOR_BGR2RGBA)
    tf.keras.utils.save_img(f"bgrem.png",rem_back_scaled)
    return rem_back_scaled






























# load and convert background to numpy array and rescale(255 for RBG images)
# background_input = load_img(args["background"])
# background_inp_img = img_to_array(background_input)
# background_inp_img /= 255

# # get dimensions of background (original image will be resized to dimensions of background image in this notebook)
# background_height = background_inp_img.shape[0]
# background_width = background_inp_img.shape[1]

# # resize the image
# resized_rem_back  = cv.resize(rem_back, (background_width,background_height))


# # create a new array which will store the final result
# output_chbg = np.zeros((background_height, background_width, 3))

# # using the following o[c] = b[c]*(1-i[t])+i[c] {where o - output image, c - channels from 1-3, i - input image with background removed, t - transparent channel}, obtain values for the final result
# output_chbg[:,:,0] = background_inp_img[:,:,0]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,0]
# output_chbg[:,:,1] = background_inp_img[:,:,1]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,1]
# output_chbg[:,:,2] = background_inp_img[:,:,2]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,2]

# # rescale
# output_chbg_scaled = Img.fromarray((output_chbg*255).astype('uint8'), 'RGB')
# out = args["input_image"].split('/')[-1].split('.')[0]

# tf.keras.utils.save_img(f"{out}bg.png",output_chbg_scaled)








