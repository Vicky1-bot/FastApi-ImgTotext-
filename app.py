from fastapi import FastAPI, Request, File, UploadFile
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import io
import cv2
import pytesseract
import configparser
from preprocess import resize,convert2Gray,threshold,cropZone

config = configparser.ConfigParser()
config.read('path.ini')
engine_path = config['global']['tesseract_engine']
path_json = config['global']['path_json']

pytesseract.pytesseract.tesseract_cmd = engine_path
custom_config = r'--oem 3 --psm 6'

class ImageType(BaseModel):
  url: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def text_img(img):
    text = pytesseract.image_to_string(img, config=custom_config, lang="eng")
    #text = pytesseract.image_to_string(image, config=custom_config, lang="mcr")
    return(text)


@app.post("/extract_text") 
async def extract_text(request: Request):
    label = ""
    if request.method == "POST":
        form = await request.form()
        # file = form["upload_file"].file
        contents = await form["upload_file"].read()
        image_stream = io.BytesIO(contents)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = resize(frame)
        frame = convert2Gray(frame)
        # frame = threshold(frame)
        ret, thresh1 = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
        crop = cropZone(thresh1)
        label = text_img(crop)

        # return {"label": label}
   
    return templates.TemplateResponse("index.html", {"request": request, "label": label})