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

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str

def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)

class Writer():
    def __init__(self, output_file, input_fps, input_framesize, input_pix_fmt,
                 input_vcodec):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt="bgr24",
    #               s='%sx%s'% ((math.ceil(input_framesize[1])*2)*0.5,(math.ceil(input_framesize[0])*2)*0.5)),
                   s= '%sx%s' % (input_framesize[1],input_framesize[0]),
                   r=input_fps)
            .output(output_file, pix_fmt=input_pix_fmt, vcodec=input_vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()

model = YOLO('yolov8n-pose.pt')
video_path = "video/test1.mp4"
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
ffprobe_result = ffprobe(video_path)
#print(ffprobe_result)
info = json.loads(ffprobe_result.json)

videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
input_fps = videoinfo["avg_frame_rate"]
# input_fps = float(input_fps[0])/float(input_fps[1])
input_pix_fmt = videoinfo["pix_fmt"]
input_vcodec = videoinfo["codec_name"]

# define a writer object to write to a movidified file
postfix = info["format"]["format_name"].split(",")[0]
output_file = ".".join(video_path.split(".")[:-1])+".processed." + postfix
writer = None
data_name=0
# 这里若cap = cv2.VideoCapture(0)
# 便是打开电脑的默认摄像头
while cap.isOpened():
    sit=0
    walk=0
    danger=0
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
                                sit+=1
                                #print("正常")
                            elif i ==1:
                                walk+=1
                                #print("跌倒")
                            elif i ==2:
                                danger+=1
                    # Let's show the predicted labels on the screen to compare with the real ones
                    #print('Predicted: ', ' ',predicted)
                    #print(type(predicted))
                    #np.save('./dataset_rnn/0/tensor_data'+str(data_name)+'.npy', data_lists1)
                    data_name+=1
            print("All person:",npkeypoints.shape[0],"  ")
            #print("sit:",sit)
            print("normal:",walk+sit)
            print("danger:",danger)
        else:
            print("All person:",0,"  ")
            #print("sit:",sit)
            print("normal:",walk+sit)
            print("danger:",danger)


        annotated_frame = results[0].plot()
        if writer is None:
            input_framesize = annotated_frame.shape[:2]
            writer = Writer(output_file, input_fps, input_framesize, input_pix_fmt,
                            input_vcodec)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        writer(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
writer.close()
cv2.destroyAllWindows()
