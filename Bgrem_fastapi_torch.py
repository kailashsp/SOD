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

import torch
import torch.nn.functional as F
import torchvision

from torchvision import transforms

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


md = U2NET()

device = 'cpu'

md = md.to(device)
chkpt = torch.load('u2net_portrait.pth')

md.load_state_dict(chkpt)

md.eval()


model = tf.keras.models.load_model('u2net_keras.h5')

# inf = model.signatures["serving_default"]
# print(inf)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def loadImage(base64str):
    "returns image in numpy array form"
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    pilimage = Image.open(bytesObj) 
    image = cv.cvtColor(np.array(pilimage),cv.IMREAD_COLOR)
    # image = cv.imread(path)
    print(f"this is the image shape{image.shape}")
    
    return base64bytes,image

    
def sharpen(img):
    " sharpens the images for better segmentation"
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv.filter2D(src=img,ddepth= -1, kernel =kernel)
    return image_sharp

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(320),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# normalize the predicted SOD probability map


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(org_img,pred):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = im.resize((org_img.shape[1],org_img.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)


    # imo.save(d_dir+'/'+image_name+'.png')

    return imo

def predict(inp_img,org_img):
    # model_name='u2net_portrait' #u2netp

    # prediction_dir = '/home/kailash/SOD/results'
    # if(not os.path.exists(prediction_dir)):
    #     os.mkdir(prediction_dir)

    model_dir = 'U2net.pth'

    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)

    net.load_state_dict(torch.load(model_dir))

    device=torch.device("cpu")
    net.to(device)
    net.eval()


    inputs_test = inp_img
    inputs_test = inputs_test.type(torch.FloatTensor)


    inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
    print(d1)
    # normalization
    pred = d1[:,0,:,:]

    pred = normPRED(pred)

    # save results to test_results folder
    seg = save_output(org_img,pred)

    del d1,d2,d3,d4,d5,d6,d7
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
    print(inp_img.shape,"shapes",a_layer.shape)

    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)

    # tf.keras.utils.save_img("rgbinp.png",rgba_inp)

    rem_back = (rgba_inp*rgba_out)
 
    # rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')
    rem_back_scaled =(rem_back*RESCALE).astype('uint8')
    rem_back_scaled= cv.cvtColor(rem_back_scaled, cv.COLOR_BGR2RGBA)
    tf.keras.utils.save_img(f"bgrem.png",rem_back_scaled)
    return rem_back_scaled




