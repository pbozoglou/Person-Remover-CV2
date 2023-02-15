import cv2
import numpy as np
import torch


def yolov5_initialize():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    with open('labels.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # set model parameters
    model.conf = 0.25 
    model.iou = 0.45 
    model.agnostic = False  
    model.multi_label = False 
    model.max_det = 1000  

    return model,classes


def yolov5_predict(img,model):

    results = model(img, size=640)

    results = model(img, augment=True)

    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    box_list = []
    score_list = []
    categories_list = []
    for i in range(len(boxes)):
        if torch.cuda.is_available():
            box_list.append([int(x) for x in boxes[i].cpu().numpy().tolist()])
            score_list.append(scores[i].cpu().numpy().tolist())
            categories_list.append(categories[i].cpu().numpy().tolist())
        else:
            box_list.append([int(x) for x in boxes[i].numpy().tolist()])
            score_list.append(scores[i].numpy().tolist())
            categories_list.append(categories[i].numpy().tolist())

    return box_list,score_list,categories_list


def person_remover(image,points):
    #image is assumed to be a cv2 numpy tensor (cv2.imread(image)) and converted rgb format.
    
    image_cp = image.copy()
    mask = image.copy()
    #mask = cv2.cvtColor(image_cp, cv2.COLOR_RGB2GRAY)
    mask[:] = 0

    num_channels = 1 if len(mask.shape) == 2 else mask.shape[2]

    for x1,y1,x2,y2 in points:
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255,) * num_channels, -1)
    
    image_cp = cv2.bitwise_or(image_cp,mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    removed = cv2.inpaint(image_cp,mask,1,cv2.INPAINT_NS)

    return removed