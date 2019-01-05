import cv2 as cv
import argparse, os
from imgseg import ImageSegmentation
from interactive import main as main_

def main(args, imgseg):
    output_file = "_mask_rcnn_out"
    if args.image != '':
        # open image file
        if not os.path.isfile(args.image):
            raise Exception("Input image file {} doesn't exist.".format(args.image))
        cap = cv.VideoCapture(args.image)
        name, ext = os.path.splitext(args.image)
    elif args.video != '':
        # open video file
        if not os.path.isfile(args.video):
            raise Exception("Input video file {} doesn't exist.".format(args.video))
        cap = cv.VideoCapture(args.video)
        name, ext = os.path.splitext(args.video)
    else:
        # webcam input
        cap = cv.VideoCapture(0)
        name, ext = "webcam", ".avi"
    
    output_file = name + output_file + ext
    if os.path.isfile(output_file):
        os.remove(output_file)

    imgseg.edit_frames(cap, output_file, (args.image!=''), args.render)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # editing params
    parser.add_argument('--image', nargs='?', type=str, default='', help='file name of image to mask')
    parser.add_argument('--video', nargs='?', type=str, default='', help='file name of video to mask')
    parser.add_argument('--conf', nargs='?', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--mask', nargs='?', type=float, default=0.3, help='mask threshold')

    # editing configs
    parser.add_argument('--render', action='store_true', default=False, help='render frame editing')
    parser.add_argument('--interactive', action='store_true', default=False, help='render frame editing')

    args = parser.parse_args()

    # model files
    model_dir = os.path.join(os.getcwd(), 'model', 'mask_rcnn_inception_v2_coco_2018_01_28')
    data_dir = os.path.join(os.getcwd(), 'data')
    classes_file = os.path.join(data_dir, 'mscoco_labels.txt')
    colors_file = os.path.join(data_dir, 'colors.txt')
    text_graph_file = os.path.join(model_dir, 'frozen_text_graph.pbtxt')
    model_weights_file = os.path.join(model_dir, 'frozen_inference_graph.pb')
    
    # load classes, colors, weights, and thresholds
    imgseg = ImageSegmentation(classes_file, colors_file, text_graph_file, model_weights_file, args.conf, args.mask)

    if args.interactive:
        main_(args, imgseg)
    else:
        main(args, imgseg)
