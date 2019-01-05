from bottle import route, request, static_file, run
from imgseg import ImageSegmentation
import cv2 as cv
import os

@route('/')
def root():
    return static_file('interactive.html', root='.')

@route('/upload', method='POST')
def do_upload():
    upload = request.files.get('upload')
    fname = os.path.abspath(upload.filename)
    name, ext = os.path.splitext(fname)
    if ext in ['.png', '.jpg', '.jpeg']:
        is_image = True
    elif ext in ['.avi', '.flv']:
        is_image = False
    else:
        raise Exception('Incompatible file type.')

    save_path = name + "_mask_rcnn_out" + ext

    # load graphics
    cap = cv.VideoCapture(fname)

    # edit graphics
    global imgseg
    imgseg.edit_frames(cap, save_path, is_image, False)

    return "File successfully saved to '{0}'.".format(save_path)

def main(args, imgseg_):
    global imgseg
    imgseg = imgseg_

    run(host='localhost', port=8080)
