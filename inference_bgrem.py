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




model = tf.keras.models.load_model('u2net_keras.h5')

# inf = model.signatures["serving_default"]
# print(inf)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#to accept an image as an input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image",help="path to the input image",required=True)

ap.add_argument("-o","--output_image",help="path to output image",default="segm.png")

ap.add_argument("-t","--threshold",help="set value to sharpen the segmentation", default=.9)

ap.add_argument("-s","--sharpen",help="shapen the input image", default=False)

ap.add_argument("-b","--background",help="new background")

args = vars(ap.parse_args())


def loadImage(path):
    "returns image in numpy array form"

    return cv.imread(path)

# def preprocess(image):
#     alpha = 10
#     beta = 10
#     image = cv.convertScaleAbs(image,alpha=alpha,beta=beta)
#     return image

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
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv.filter2D(src=img,ddepth= -1, kernel =kernel)
    return image_sharp

def predict(img):
    "returns the class of prediction"

    # pred = inf(tf.constant(img))

    pred = model.predict(img)
    pred = np.array(pred[0])
    # pred = list(pred.values())[0]

    predict_img = np.squeeze(pred, axis=0) #remove batch axis (1,256,256,1) => (256,256,1)
    predict_img.shape
    print(predict_img.shape)
    data = np.moveaxis(predict_img, 0, 2)
    print(data.shape)
    data = np.expand_dims(cv.resize(data, (w, h)),axis=2)
    print(data.shape)
    tf.keras.utils.save_img(f'{args["output_image"]}',data)


i_image = args["input_image"]


image = loadImage(i_image)

h = image.shape[0]
w = image.shape[1]


# sharpen image
if args['sharpen']:
    image = sharpen(image)


reimage = normalize(image)

# print(image.shape)

predict(reimage)

output = load_img(f'{args["output_image"]}')

# BACKGROUND REMOVAL
def back_rem(output):
    


    RESCALE = 255
    out_img = img_to_array(output)
    out_img /= RESCALE


    THRESHOLD = float(args["threshold"])

    # refine the output
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    shape = out_img.shape
    print(shape)
    print('+++++++++++++++++++++++++++++++')
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)
    tf.keras.utils.save_img("rgbaout.png",rgba_out)
   
    print(rgba_out.shape)
    input = load_img(i_image)
    inp_img = img_to_array(input)
    # inp_img= cv.resize(inp_img, (320, 320))

    inp_img /= RESCALE


    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)
    tf.keras.utils.save_img("rgbinp.png",rgba_inp)

    print(rgba_inp.shape)
    rem_back = (rgba_inp*rgba_out)
    print(rem_back.shape)
    print("+++++++++++++++++++++++++")
    # rem_back = cv.resize(rem_back,(w,h))

    # blur = cv.GaussianBlur(rem_back,(5,5), sigmaX=0, sigmaY=0) 

    # rem_back = np.uint8(rem_back)
    # blur = cv.Canny(image=rem_back, threshold1=1, threshold2=200)

    # result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))

    # filt = cv.GaussianBlur(rem_back,(3,3),0)

    rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')


    bgrem = args["input_image"].split('/')[-1].split('.')[0]

    tf.keras.utils.save_img(f"{bgrem}bgrem.png",rem_back_scaled)

    return rem_back

rem_back = back_rem(output)

# load and convert background to numpy array and rescale(255 for RBG images)
background_input = load_img(args["background"])
background_inp_img = img_to_array(background_input)
background_inp_img /= 255

# get dimensions of background (original image will be resized to dimensions of background image in this notebook)
background_height = background_inp_img.shape[0]
background_width = background_inp_img.shape[1]

# resize the image
resized_rem_back  = cv.resize(rem_back, (background_width,background_height))


# create a new array which will store the final result
output_chbg = np.zeros((background_height, background_width, 3))

# using the following o[c] = b[c]*(1-i[t])+i[c] {where o - output image, c - channels from 1-3, i - input image with background removed, t - transparent channel}, obtain values for the final result
output_chbg[:,:,0] = background_inp_img[:,:,0]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,0]
output_chbg[:,:,1] = background_inp_img[:,:,1]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,1]
output_chbg[:,:,2] = background_inp_img[:,:,2]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,2]

# rescale
output_chbg_scaled = Img.fromarray((output_chbg*255).astype('uint8'), 'RGB')
out = args["input_image"].split('/')[-1].split('.')[0]

tf.keras.utils.save_img(f"{out}bg.png",output_chbg_scaled)








