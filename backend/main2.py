# Importing Necessary modules
from fastapi import FastAPI, File, UploadFile
import base64
import cv2 as cv
from pydantic import BaseModel
import json
from io import BytesIO
from starlette.responses import StreamingResponse
from Bgrem_fastapi_torch import loadImage, predict , back_rem , transform_image
 
# Declaring our FastAPI instance
app = FastAPI()

def im2json(im):
    """Convert a Numpy array to JSON string"""
    _, imdata = cv.imencode('.JPG',im)
    jstr = json.dumps({"image": base64.b64encode(imdata).decode('ascii')})
    return jstr

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in {"jpg", "jpeg", "png"}
    print(file.filename)
    if not extension:
        return "Image must be jpg or png format!"

    content = await file.read()
    base64str = base64.b64encode(content).decode("utf-8")

    b_image, i_image  = loadImage(base64str)

    image =  transform_image(b_image)
    segm = predict(inp_img=image, org_img =i_image) 

    rem_bg = back_rem(segm, i_image)
    
    # img_str = cv.imencode('.png', rem_bg)[1]
    # return StreamingResponse(BytesIO(img_str.tobytes()), media_type="image/png")
    return im2json(rem_bg)


