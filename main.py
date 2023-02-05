# Importing Necessary modules
from fastapi import FastAPI, File, UploadFile
import base64
import cv2 as cv
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from Bgrem_fasapi import loadImage, sharpen,  normalize , predict , back_rem
 
# Declaring our FastAPI instance
app = FastAPI()
 
async def bgrem(img):
    for i in range(10):
        yield b"some fake video bytes"

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in {"jpg", "jpeg", "png"}
    print(file.filename)
    if not extension:
        return "Image must be jpg or png format!"

    with open(f"images/{file.filename}", "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")

    org_image , h , w = loadImage(base64str)
    image = sharpen(org_image)
    image = normalize(image)
    segm = predict(image, h, w) 

    rem_bg = back_rem(segm, org_image)
    
    # filtered_image = BytesIO(rem_bg)
    # rem_bg.save(filtered_image, "JPEG")
    # filtered_image.seek(0)
    img_str = cv.imencode('.png', rem_bg)[1]
    # encoded_img = base64.b64encode(img_str)
    # en_image = BytesIO(rem_bg)
    # # en_image.save(en_image, "JPEG")
    # en_image.seek(0)

    return StreamingResponse(BytesIO(img_str.tobytes()), media_type="image/png")





