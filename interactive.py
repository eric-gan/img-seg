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

    # save to temporary file so opencv can access it
    fname = os.path.abspath(upload.filename)
    upload.save(fname, overwrite=True)

    name, ext = os.path.splitext(fname)
    if ext in ['.png', '.jpg', '.jpeg']:
        is_image = True
    elif ext in ['.avi']:
        is_image = False
    else:
        raise Exception('Incompatible file type.')

    output_file = name + "_mask_rcnn_out" + ext

    # load and edit graphics
    cap = cv.VideoCapture(fname)
    global imgseg
    imgseg.edit_frames(cap, output_file, is_image, False)

    # delete temporary file
    os.remove(fname)

    return "File successfully saved to '{0}'.".format(save_path)

def main(args, imgseg_):
    global imgseg
    imgseg = imgseg_

    run(host='localhost', port=8080)
