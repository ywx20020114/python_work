import os
import shutil
import numpy as np
import apps.utils.data_load
import cv2 as cv
import apps.PCA.PCA
import apps.SVM.svm

from collections import Counter
import mediapipe as mp


# 导入人脸识别模块
mpFace = mp.solutions.face_detection
# 导入绘图模块
mpDraw = mp.solutions.drawing_utils
# 自定义人脸识别方法，最小的人脸检测置信度0.5
faceDetection = mpFace.FaceDetection(min_detection_confidence=0.5)

# 启动摄像头，实时进行人脸识别
# 参数 ： 文件路径名称
# 返回值 ：出现次数最多的人名 以及照片

def FaceRecognize(data):


    #保存检测出的名字
    name_list = []
    top_name = ''

    # 加载训练数据
    X, y, names = apps.utils.data_load.LoadData(data)

    model = apps.SVM.svm.svc(data)


    # 打开摄像头
    camera = cv.VideoCapture(0)
    # cv.namedWindow('Face')
    count = 0

    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        # 判断图片读取成功
        if ret:
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 人脸检测
            results = faceDetection.process(frame)
            for index, detection in enumerate(results.detections):
                # 遍历每一帧图像并打印结果
                # print(index, detection)
                # 每帧图像返回一次是人脸的几率，以及识别框的xywh，后续返回关键点的xy坐标
                print(detection.score)  # 是人脸的的可能性
                print(detection.location_data.relative_bounding_box)  # 识别框的xywh

                # 设置一个边界框，接收所有的框的xywh及关键点信息
                bboxC = detection.location_data.relative_bounding_box

                # 接收每一帧图像的宽、高、通道数
                ih, iw, ic = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            #x, y, w, h = my_predict.predict(yolo,frame)

            # x = int(x)
            # y = int(y)
            # w = int(w)
            # h = int(h)
            # 在原图像上绘制矩形
            if w == -1:
                continue
            frame = cv.rectangle(frame, (x,y),(x+w,y+h), (255, 0, 0), 2)
            roi_gray = gray_img[y:y+h,x:x+w]

            # 用于返回头像照片
            temp = frame[y:y+h,x:w+w]

            # 宽92 高112
            roi_gray = cv.resize(roi_gray, (92, 112), interpolation=cv.INTER_LINEAR)


            # 对图片进行PCA降维  (x,y分别降维到 50个特征点)
            roi_gray = apps.PCA.PCA.PCA_Data(roi_gray)
            roi_gray = np.transpose(roi_gray)
            roi_gray = apps.PCA.PCA.PCA_Data(roi_gray)
            roi_gray = np.transpose(roi_gray)

            roi_gray = roi_gray.ravel()
            roi_gray = roi_gray.reshape(1,-1)
            # print(roi_gray)

            # 进行预测
            label = model.predict(roi_gray)
            # print(label)
            # print(names)
            # print('Label:%s,confidence:%.2f' % (params[0], params[1]))
            # 将预测的实时名字写入图片中
            cv.putText(frame, names[label[0]], (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            # 将所有出现的人名 照片放入list当中
            name_list.append(names[label[0]])


            # 找到出现最多次的人名
            name_counts = Counter(name_list)
            top_name = name_counts.most_common(1)

            name_path = 'apps/Name'
            if count == 200:
                # 如果路径存在则删除
                if os.path.isdir(name_path):
                    shutil.rmtree(name_path)  # 递归删除文件夹

                # 如果没有文件夹  创建文件
                if not os.path.exists(name_path):
                    os.mkdir(name_path)

                # 创建一个txt文件，文件名为temp,并向文件写入msg
                full_path = name_path + '\\temp' + '.txt'  # 也可以创建一个.doc的word文档
                file = open(full_path, 'w')
                file.write(top_name[0][0])  # msg也就是下面的Hello world!
                print(top_name[0][0])
                file.close()
                return
            count += 1
            image = cv.imencode('.jpg', frame)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')



    #     cv.imshow('Face', frame)
    #     # 如果按下q键则退出
    #     if cv.waitKey(100) & 0xff == ord('q'):
    #         break
    # camera.release()
    # cv.destroyAllWindows()



