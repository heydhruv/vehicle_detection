import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn

from base.services.trackbleobject import TrackableObject
from base.services.utils.general import set_logging
from base.services.models.common import DetectMultiBackend
from base.services.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from base.services.utils.general import (LOGGER, Profile, check_file, check_img_size,
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from base.services.utils.plots import Annotator, colors, save_one_box
from base.services.utils.torch_utils import select_device, time_sync

from base.com.dao.video_dao import VideoDAO
from base.com.vo.video_vo import VideoVO
from datetime import date
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

#---------------Object Tracking---------------
import skimage
from base.services.sort import *


#-----------Object Blurring-------------------
blurratio = 40


#.................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# -----------------------
trackers = []
trackableObjects = {}

totalDown = 0
totalUp = 0
# -----------------------


def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, 
                names=None, color_box=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255,191,0),-1)
    return img
#..............................................................................


@torch.no_grad()
def detect(source,weights=r'base/services/yolov5s.pt',
        # source=r'C:\Users\Dhruv\Desktop\yolov5-object-tracking\vehicle_video_1.mp4',
        data=r'base/services/data/coco128.yaml',
        imgsz=(640, 640),conf_thres=0.4,iou_thres=0.45,
        max_det=1000, device='cpu',  view_img=True,
        save_txt=False, save_conf=False, save_crop=False, 
        nosave=False, classes=None,  agnostic_nms=False,  
        augment=False, visualize=False,  update=False,  
        project=r'base/static/adminResources/output_video',  name='',
        exist_ok=False, line_thickness=2,hide_labels=False,  
        hide_conf=False,half=False,dnn=False,display_labels=False,
        blur_obj=False,color_box = False,):

    print("><><><><><><>")
    save_img = not nosave and not source.endswith('.txt') 
    save_path = ""
    print("<<<<<<<<<<<")
    #.... Initialize SORT .... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 
    track_color_id = 0
    #......................... 
    
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(project) / name
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1 
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    t0 = time.time()
    
    dt, seen = [0.0, 0.0, 0.0], 0
    # ----------------------------------
    # multi onject up down variables
    listDet = ['person', 'bicycle', 'car', 'motorcycle']
    rects = []
    labelObj = []


    totalDownCar = 0
    totalDownMotor = 0
    totalDownBus = 0
    totalDownTruck = 0


    totalUpCar = 0
    totalUpMotor = 0
    totalUpBus = 0
    totalUpTruck = 0
    # ----------------------------------
    for path, im, im0s, vid_cap, s in dataset:
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            middle = im0.shape[0] // 2
            height = im0.shape[0]
            width = im0.shape[1]
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if blur_obj:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                        blur = cv2.blur(crop_obj,(blurratio,blurratio))
                        im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                    else:
                        continue
                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                              np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                 
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()
                

                
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    for cord in tracked_dets:
                        # print("cord ->", cord)
                        x_c = (cord[0] + [2]) / 2
                        print(x_c)
                        y_c = (cord[1] + cord[3]) / 2
                        print(y_c)
                        to = trackableObjects.get(cord[8], None)
                        if to is None:
                            to = TrackableObject(cord[8], (x_c, y_c))
                            print("TOOOO", to)
                        else:
                            y = [c[1] for c in to.centroids]
                            direction = y_c - np.mean(y)
                            to.centroids.append((x_c, y_c))
                            if not to.counted:
                                idx = int(cord[4])
                                print("idx", type(idx))
                                if direction < 0 and middle > y_c > height // 2.3:
                                    if (idx == 2):
                                        totalUpCar += 1
                                        to.counted = True
                                    elif (idx == 3 or idx == 1):
                                        totalUpMotor += 1
                                        to.counted = True
                                    elif (idx == 5):
                                        totalUpBus += 1
                                        to.counted = True
                                    elif (idx == 7):
                                        totalUpTruck += 1
                                        to.counted = True
                                    totalUp=totalUpBus+totalUpCar+totalUpMotor+totalUpTruck
                                elif direction > 0 and middle < y_c < height // 1.8:
                                    if (idx == 2):
                                        totalDownCar += 1
                                        to.counted = True
                                    elif (idx == 3 or idx == 1):
                                        totalDownMotor += 1
                                        to.counted = True
                                    elif (idx == 5):
                                        totalDownBus += 1
                                        to.counted = True
                                    elif (idx == 7):
                                        totalDownTruck += 1
                                        to.counted = True
                                    totalDown=totalDownBus+totalDownCar+totalDownMotor+totalDownTruck
                                    # empty_space_car=totalDownCar-totalUpCar
                                    # empty_space_moter=totalDownMotor-totalUpMotor
                                    # empty_space_bus=totalDownBus-totalUpBus
                                    # empty_space_truck=totalDownTruck-totalUpTruck
                        trackableObjects[cord[8]] = to

                    
                    draw_boxes(im0, bbox_xyxy, identities, categories, names,color_box)
            start_point = (0, int(height // 2.3))
            start_point_middle = (0, int(middle))
            start_point_below = (0, int(height // 1.8))

            end_point = (int(im0.shape[1]), int(height // 2.3))
            end_point_middle = (int(im0.shape[1]), int(middle))
            end_point_below = (int(im0.shape[1]), int(height // 1.8))

            color1 = (0,0,255)
            thickness = 4
            # cv2.line(im0,(0,int(height//2)),(int(width//2.4),0),thickness)
            cv2.line(im0, start_point, end_point,color1, thickness)
            cv2.line(im0, start_point_middle, end_point_middle,color1, thickness)
            cv2.line(im0, start_point_below, end_point_below,color1, thickness)

            color = (0, 255, 0)
            cv2.putText(im0, 'Down car : ' + str(totalDownCar), (int(width * 0.6), int(height * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Down motorcycle : ' + str(totalDownMotor), (int(width * 0.6), int(height * 0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Down bus : ' + str(totalDownBus), (int(width * 0.6), int(height * 0.25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Down truck : ' + str(totalDownTruck), (int(width * 0.6), int(height * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # cv2.putText(im0, 'Total Down : ' + str(totalDown), (int(width * 0.02), int(height * 0.4)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Up car : ' + str(totalUpCar), (int(width * 0.02), int(height * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Up motorcycle : ' + str(totalUpMotor), (int(width * 0.02), int(height * 0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Up bus : ' + str(totalUpBus), (int(width * 0.02), int(height * 0.25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Up truck : ' + str(totalUpTruck), (int(width * 0.02), int(height * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)                    

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1) 
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  
                        if vid_cap: 
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            src=os.path.split(source)
                            ip_name=src[1].split(".")
                            video_save_path=project+str(ip_name[0]+".mp4")
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'h264'), fps, (w, h))
                    vid_writer.write(im0)
        print("Frame Processing!")
    print("Video Exported Success")

    if update:
        strip_optimizer(weights)
    
    if vid_cap:

        vid_cap.release()

    return save_path,totalDown,totalUp
# def parse_opt():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='show results', default=True)
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # parser.add_argument('--blur-obj', action='store_true', help='Blur Detected Objects')
    # parser.add_argument('--color-box', action='store_true', help='Change color of every box and track')
    # opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    # return opt


# def main():
#     check_requirements(exclude=('tensorboard', 'thop'))
#     detect(**vars())
#
#
# if __name__ == "__main__":
#
#     main()
