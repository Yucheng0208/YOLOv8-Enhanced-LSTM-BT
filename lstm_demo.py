from ultralytics import YOLO
from lstm import LSTM_Model
import numpy as np
import torch
model = YOLO('yolov8n-pose.pt')

def testBatch(lstmdata):
    import torchvision
    '''
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                               for j in range(batch_size)))
    '''
    # Let's see what if the model identifiers the  labels of those example

    if torch.cuda.is_available():
        lstmdata = torch.tensor(lstmdata).cuda()
    else:
        lstmdata = torch.tensor(lstmdata)
        #print("請使用GPU")

    outputs = LSTM_Model(lstmdata.float())

    if torch.cuda.is_available():
        outputs = outputs.cpu()
    else:
        print("使用CPU")

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    predicted = np.array(predicted)
    print(predicted)
    for i in predicted:
        if i==1:
            print("跌倒")
        if i==0:
            print("正常")
    # Let's show the predicted labels on the screen to compare with the real ones
    #print('Predicted: ', ' ',predicted)
    #print(type(predicted))
'''
source = 'path/to/image.jpg'
source = 'screen'
source = 'https://ultralytics.com/images/b.jpg'
source = Image.open('path/to/image.jpg')
source = cv2.imread('path/to/image.jpg')
source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')
source = torch.rand(1, 3, 640, 640, dtype=torch.float32)
# Define a path to a CSV file with images, URLs, videos and directories
source = 'path/to/file.csv'
source = 'path/to/video.mp4'
# Define path to directory containing images and videos for inference
source = 'path/to/dir'
source = 'path/to/dir/**/*.jpg'
source = 'https://youtu.be/LNwODJXt4'
source = 'rtsp://example.com/media.mp4'  # RTSP, RTMP, TCP or IP streaming address
'''

source = 'images/4.jpg'
results = model(source)
result = model.predict(source)[0]
data_name=0
lstm_path = "LSTM_Model.pth"
LSTM_Model=LSTM_Model()

LSTM_Model.load_state_dict(torch.load(lstm_path))

if torch.cuda.is_available():
    LSTM_Model = LSTM_Model.cuda()
else:
    LSTM_Model = LSTM_Model
    #print("請使用GPU")
#print(data_teat)
if result.keypoints.conf != None:
    #keypoints = result.keypoints.data.tolist()
    keypoints = result.keypoints.xyn.tolist()
    confs = result.boxes.conf.tolist()
    keypointsconf = result.keypoints.conf.tolist()
    #print(result.boxes)
    #print(result.keypoints)
    '''
    for i, kp in enumerate(keypoints):
        x = int(kp[0])
        y = int(kp[1])
    '''
    #print(11111)
    #print(keypoints)
    #print(confs)
    npconfs = np.array(confs)
    npkeypoints = np.array(keypoints)
    npkeypointsconf = np.array(keypointsconf)
    #print(npkeypoints.shape)
    #print(11111)

    for a in range(npkeypoints.shape[0]):
        data_listx=[]
        data_listy=[]
        if npconfs[a] >= 0.5:
            for b in range(17):
                if npkeypointsconf[a][b] >= 0.4:
                    data_listx.append(npkeypoints[a][b][0])
                    data_listy.append(npkeypoints[a][b][1])
                else:
                    data_listx.append(-1)
                    data_listy.append(-1)
            data_list=np.vstack([data_listx,data_listy])
            print(data_list)
            data_teat=np.reshape(data_list,(-1,1,34))
            # Test with batch of images
            testBatch(data_teat)
            #np.save('./dataset/0/tensor_data'+str(data_name)+'.npy', data_list)
            data_name+=1
from PIL import Image
for r in results:
    im_array = r.plot()
    #print(11111)
    #print((im_array[..., ::-1]).shape)
    #print(22222)
    im = Image.fromarray(im_array[..., ::-1])
    im.show()  # show image
    im.save('results3.jpg')
