import cv2
def resize(img):
    return cv2.resize(img, (1200,600))

def convert2Gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def threshold(img):
    return cv2.threshold(img,60,255,cv2.THRESH_BINARY);
#ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
def cropZone(img):
    return (img[176:232,100:750])

#  , file: bytes = File(...)