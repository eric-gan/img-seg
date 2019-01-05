# Image Segmentation using Mask-RCNN
The model used is a TensorFlow Mask-RCNN with an InceptionV2 backbone (fastest among of ResNet50, ResNet101, and Inception-ResnetV2) trained on the MSCOCO dataset. The model can be ported from [here](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz). Inference works on CPU.

## Usage
The inference script only uses OpenCV, NumPy, and Tqdm for progress tracking. However, interactive image segmentation in browser requires Bottle.
```
# dependencies
pip3 install -y -r requirements.txt

# model, labels
sh download.sh
```

## Docker
To build and run a Docker container:
```
docker build -t img-seg .
docker run -it -v $PWD:/mnt img-seg
```

## Inference
During inference, boxes and masks are drawn on the input image, or on each frame of the input video. If no image or video is specified, a video is generate through the webcam. To run inference locally:
```
python3 run.py \
    --image /path/to/image
    --video /path/to/video
    --conf 0.5
    --mask 0.3
```
Additionally, the `--render` and `--interactive` flags can be used to render each frame and run uploading in browser, respectively.
