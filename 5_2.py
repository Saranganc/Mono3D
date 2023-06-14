import os
import time
import cv2
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from lib.DataUtils import *
from lib.Utils import *
from lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
from filterpy.kalman import KalmanFilter

def main():

    bins_no = 2

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    weight_list = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(weight_list) == 0:
        print('We could not find any model weight to load, please train the model first!')
        exit()
    else:
        print('Using model weights : %s'%weight_list[-1])
        my_vgg = models.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=bins_no).to(device)
        if use_cuda: 
            checkpoint = torch.load(weights_path + '/%s'%weight_list[-1])
        else: 
            checkpoint = torch.load(weights_path + '/%s'%weight_list[-1],map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Load Yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(bins_no)

    image_dir = FLAGS.val_img_path
    cal_dir = FLAGS.calb_path

    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file instead of per image calibration
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"
    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'
   
    try:
        ids = [x.split('.')[0][-6:] for x in sorted(glob.glob(img_path+'/*.png'))]
    except:
        print("\nError: There are no images in %s"%img_path)
        exit()

    # Open input video file
    video_in = cv2.VideoCapture('Kitti/test1.mp4')

    # Get video properties
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    # Create output video writer
    video_out = cv2.VideoWriter('Kitti/out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # kalman_filters = []

    dt = 1.0/fps
    v = dt*2  # 2 is good
    a = 0.5*dt**2 # 0.5*dt**2 is good

    # for _ in range(num_frames):
        
    #     kalman_filters.append(kalman)

    for id in range(num_frames):
        start_time = time.time()
        ret, frame = video_in.read()
        img_file = img_path + str(id) + ".png"

        # Read in image and make copy
        truth_img = frame
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        
        # Run Detection on yolo
        detections = yolo.detect(yolo_img)

        # For each 2D Detection
        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue
            # To catch errors should there be an invalid 2D detection
            try:
                object = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = object.theta_ray
            input_img = object.img
            proj_matrix = object.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224]).to(device)
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location, X = calc_location(dim, proj_matrix, box_2d, alpha, theta_ray)

            kalman = cv2.KalmanFilter(9, 3, 0)

            kalman.measurementMatrix = np.array([
                    [1, 0, 0, v, 0, 0, a, 0, 0],
                    [0, 1, 0, 0, v, 0, 0, a, 0],
                    [0, 0, 1, 0, 0, v, 0, 0, a]
                ],np.float32)

            kalman.transitionMatrix = np.array([
                    [1, 0, 0, v, 0, 0, a, 0, 0],
                    [0, 1, 0, 0, v, 0, 0, a, 0],
                    [0, 0, 1, 0, 0, v, 0, 0, a],
                    [0, 0, 0, 1, 0, 0, v, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, v, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, v],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]
                ],np.float32)

            kalman.processNoiseCov = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]
                ],np.float32) * 0.007

            kalman.measurementNoiseCov = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0 ,1]
                ],np.float32) * 0.1

            kalman.statePre = np.array([
                                            [np.float32(location[0])], [np.float32(location[1])], [np.float32(location[2])]
                                        , [np.float32(location[0])], [np.float32(location[1])], [np.float32(location[2])]
                                        , [np.float32(location[0])], [np.float32(location[1])], [np.float32(location[2])]
                                    ])

            # plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            
            print('Estimated pose: %s'%location)
            # Create the measurement
            measurement = np.array([
                [np.float32(location[0])],
                [np.float32(location[1])],
                [np.float32(location[2])]
            ])
            kalman.correct(measurement)
            tp = kalman.predict()

            # Update the estimated pose with the corrected values
            location[0] = tp[0].item()
            location[1] = tp[1].item()
            location[2] = tp[2].item()

            print('Corrected pose: %s'%location)

            plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, color=cv_colors.RED.value, location=location)
            # plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, color=cv_colors.YELLOW.value)

            

        video_out.write(img)
        print('Frame processed in {:.5f} seconds'.format(time.time() - start_time))

    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--val-img-path", default="Kitti/validation/image_2/",
                        help="Please specify the path to the images you wish to evaluate on.")

    parser.add_argument("--calb-path", default="Kitti/camera_cal/",
                        help="Please specify the path containing camera calibration obtained from KITTI")

    parser.add_argument("--show-2D", action="store_true",
                        help="Shows the 2D BoundingBox detections of the object detection model on a separate image")

    FLAGS = parser.parse_args()

    main()
