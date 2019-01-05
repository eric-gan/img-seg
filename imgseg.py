import cv2 as cv
import numpy as np
from tqdm import tqdm

class ImageSegmentation(object):
    def __init__(self, classes_file, colors_file, text_graph_file, model_weights_file, conf, mask):
        self._init_classes(classes_file)
        self._init_colors(colors_file)
        self._init_mrcnn(text_graph_file, model_weights_file)
        self.conf = conf
        self.mask = mask

    def _init_classes(self, classes_file):
        """Initialize all class names from MSCOCO dataset."""
        classes = None
        with open(classes_file, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        self.classes = classes

    def _init_colors(self, colors_file):
        """Initialize all mask colors for MSCOCO objects."""
        with open(colors_file, 'rt') as f:
            colors_str = f.read().rstrip('\n').split('\n')

        colors = []
        for i in range(len(colors_str)):
            rgb = colors_str[i].split(' ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            colors.append(color)
        self.colors = colors

    def _init_mrcnn(self, text_graph_file, model_weights_file):
        """Load the TensorFlow Mask RCNN model."""
        model = cv.dnn.readNetFromTensorflow(model_weights_file, text_graph_file)
        model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.model = model

    def draw_box(self, frame, cid, conf, left, top, right, bottom, cmask):
        """Draw bounding box with object mask."""
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        # print class label
        label = '%.2f' % conf
        if self.classes:
            assert(cid < len(self.classes))
            label = '%s:%s' % (self.classes[cid], label)
        
        # display label at the top of the bounding box
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top + base_line), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    
        # resize the mask, threshold, color and apply it on the image
        cmask = cv.resize(cmask, (right - left + 1, bottom - top + 1))
        mask = (cmask > self.mask)
        roi = frame[top:bottom+1, left:right+1][mask]
    
        color = self.colors[cid%len(self.colors)]
        # comment the above line and uncomment the two lines below to generate different instance colors
        # cidx = random.randint(0, len(self.colors)-1)
        # color = self.colors[cidx]
    
        frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
    
        # draw the contours on the image
        mask = mask.astype(np.uint8)
        im2, contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

        return frame

    def postprocess(self, frame, boxes, masks):
        """
        Shape of masks is (N, C, H, W):
        N - number of detected boxes
        C - number of classes (excluding background)
        H, W - segmentation shape
        """
        n_classes = masks.shape[1]
        n_detections = boxes.shape[2]
        
        h = frame.shape[0]
        w = frame.shape[1]
        
        for i in range(n_detections):
            box = boxes[0, 0, i]
            mask = masks[i]
            conf = box[2]
            if conf > self.conf:
                cid = int(box[1])
                
                # extract the bounding box
                left = int(w * box[3])
                top = int(h * box[4])
                right = int(w * box[5])
                bottom = int(h * box[6])
                
                left = max(0, min(left, w - 1))
                top = max(0, min(top, h - 1))
                right = max(0, min(right, w - 1))
                bottom = max(0, min(bottom, h - 1))
                
                # extract the mask for the object
                cmask = mask[cid]
                
                # draw bounding box, colorize and show the mask on the image
                frame = self.draw_box(frame, cid, conf, left, top, right, bottom, cmask)

        return frame

    def edit_frames(self, cap, output_file, is_image, render):
        if not is_image:
            vid_writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M','J','P','G'), 28, 
                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        pbar = tqdm()
        while cv.waitKey(1) < 0:
            # get frame from the video
            has_frame, frame = cap.read()
            
            # stop the program if end of video
            if not has_frame:
                print("Done processing! Output is stored as {}.".format(output_file))
                cv.waitKey(3000)
                break
        
            # create a 4D blob from a frame
            blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

            # set the input to the model
            self.model.setInput(blob)
        
            # run the forward pass to get output from the output layers
            boxes, masks = self.model.forward(['detection_out_final', 'detection_masks'])

            # extract the bounding box and mask for each of the detected objects
            frame = self.postprocess(frame, boxes, masks)
        
            # put efficiency information
            t, _ = self.model.getPerfProfile()
            label = 'Mask-RCNN : Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
            # write the frame with the detection boxes
            if is_image:
                cv.imwrite(output_file, frame.astype(np.uint8))
            else:
                vid_writer.write(frame.astype(np.uint8))
        
            if render:
                cv.imshow("Window", frame)
            
            pbar.update(1)
        
        pbar.close()