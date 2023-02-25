# frontend/main.py

import requests
import io
import json
import base64
from io import BytesIO
import streamlit as st
from PIL import Image

# STYLES = {
#     "candy": "candy",
#     "composition 6": "composition_vii",
#     "feathers": "feathers",
#     "la_muse": "la_muse",
#     "mosaic": "mosaic",
#     "starry night": "starry_night",
#     "the scream": "the_scream",
#     "the wave": "the_wave",
#     "udnie": "udnie",
# }

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Image background removal")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays the select widget for the styles
# style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

def json2im(jstr):
    load = json.loads(jstr)
    imdata = base64.b64decode(load['image'])
    im = Image.open(BytesIO(imdata))
    return im

# displays a button
if st.button("remove background"):
    if image is not None:
        files = {"file": image}

        image = requests.post(f"http://backend:4000/predict/", files=files)
        img = json2im(image.json())
        st.image(img, width=500,output_format="PNG")