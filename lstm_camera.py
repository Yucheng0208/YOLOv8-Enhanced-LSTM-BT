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


hmodel=YOLO('hemlet.pt')


model = YOLO('yolov8n-pose.pt')
video_path = 0
cap = cv2.VideoCapture(video_path)
lstm_path = "LSTM_Model2.pth"
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

        resultss=hmodel(frame) #處理預測有無頭盔

        results = model(frame)
        result = model.predict(frame)[0]

        # ---------------------------------------------------------------------------------------------------------------
        # 偵測頭盔結果繪製
        helmet_annotated_frame = frame.copy()  # 確保不改動原始影像
        if resultss:
            print(resultss[0].boxes)  # 確認是否有偵測框
            for box in resultss[0].boxes:  # 取出偵測框
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 偵測框座標
                conf = box.conf[0]  # 信心度
                # 修改分類邏輯
                label = "Helmet" if box.cls[0] == 1 else "No Helmet"  # 索引 1 是有帽子，0 是沒帽子
                color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)  # 綠色表示有頭盔，紅色表示無頭盔
                # 繪製框和文字
                cv2.rectangle(helmet_annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    helmet_annotated_frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        else:
            print("No helmet detected")

        # **定義 annotated_frame** (人體偵測結果繪製)
        annotated_frame = results[0].plot()

        # 調整透明度並合成畫面
        combined_frame = cv2.addWeighted(annotated_frame, 0.7, helmet_annotated_frame, 0.5, 0)

        # 顯示畫面
        cv2.imshow("YOLOv8 Helmet and Pose Detection", combined_frame)

        # *----------------------------------------------------------------------------------


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
        #cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()