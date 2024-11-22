from ultralytics import YOLO
from typing import NamedTuple
from lstm import LSTM_Model
import torchvision
import torch
import subprocess
import numpy as np
import cv2
import ffmpeg
import json
import os

model = YOLO('yolov8n-pose.pt')
video_path = 0
cap = cv2.VideoCapture(video_path)
lstm_path = "LSTM_Model.pth"
LSTM_Model=LSTM_Model()
LSTM_Model.load_state_dict(torch.load(lstm_path))
if torch.cuda.is_available():
    LSTM_Model = LSTM_Model.cuda()
else:
    LSTM_Model = LSTM_Model
    #print("請使用GPU")
# get video file info
print(video_path)
data_name=0
# 这里若cap = cv2.VideoCapture(0)
# 便是打开电脑的默认摄像头
while cap.isOpened():
    normal=0
    fall=0
    success, frame = cap.read()
    if success:
        results = model(frame)
        result = model.predict(frame)[0]
        if result.keypoints.conf != None:
            #keypoints = result.keypoints.data.tolist()
            #print(torch.tensor([]))
            #print(result.keypoints.conf)
            #print(result.boxes.conf)
            #print(result.keypoints.xyn)
            keypoints = result.keypoints.xyn.tolist()
            confs = result.boxes.conf.tolist()
            keypointsconf = result.keypoints.conf.tolist()
            #print(result.boxes)
            #print(result.keypoints)
            #print(keypoints)
            #print(confs)

            npconfs = np.array(confs)
            npkeypoints = np.array(keypoints)
            npkeypointsconf = np.array(keypointsconf)
            #print(npkeypoints.shape)

            for a in range(npkeypoints.shape[0]):
                data_listx=[]
                data_listy=[]
                if npconfs[a] >= 0.6:
                    for b in range(17):
                        if npkeypointsconf[a][b] >= 0.5:
                            data_listx.append(npkeypoints[a][b][0])
                            data_listy.append(npkeypoints[a][b][1])
                        else:
                            data_listx.append(-1)
                            data_listy.append(-1)
                    data_list=np.vstack([data_listx,data_listy])
                    print(data_list)
                    #np.save('./dataset/1/tensor_data'+str(data_name)+'.npy', data_list)
                    data_teat=np.reshape(data_list,(-1,1,34))
                    # Test with batch of images
                    # Let's see what if the model identifiers the  labels of those example
                    if torch.cuda.is_available():
                        lstmdata = torch.tensor(data_teat).cuda()
                    else:
                        lstmdata = torch.tensor(data_teat)
                        #print("請使用GPU")
                    outputs = LSTM_Model(lstmdata.float())
                    if torch.cuda.is_available():
                        outputs = outputs.cpu()
                    #print(outputs)
                    # We got the probability for every 10 labels. The highest (max) probability should be correct label
                    ef, predicted = torch.max(outputs,1)
                    #print(predicted)
                    #print(ef)
                    ef = ef.detach().numpy()
                    predicted = np.array(predicted)
                    #print(type(predicted))
                    #print(ef)
                    n=[predicted[0]]
                    if np.count_nonzero(data_list == -1) <= 34:
                        for i in n:
                            if i==0:
                                normal+=1
                                #print("正常")
                            elif i ==1:
                                fall+=1
                                #print("跌倒")
                    # Let's show the predicted labels on the screen to compare with the real ones
                    #print('Predicted: ', ' ',predicted)
                    #print(type(predicted))
                    #np.save('./dataset_rnn/0/tensor_data'+str(data_name)+'.npy', data_lists1)
                    data_name+=1
            print("總人數:",npkeypoints.shape[0],"  ")
            print("符合正常人數:",normal)
            print("跌倒危險人數:",fall)
        else:
            print("總人數:",0,"  ")
            print("符合正常人數:",normal)
            print("跌倒危險人數:",fall)


        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
