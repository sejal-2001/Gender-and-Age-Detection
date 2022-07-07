# importing libraries

import cv2
import math
import argparse


# detect the face
def highlightFace(net, frame, conf_threshold=0.7):
    # copy a frame of the image to frameOpencvDnn
    frameOpencvDnn = frame.copy()
    # Image window height
    frameHeight = frameOpencvDnn.shape[0]
    # Width of the picture window
    frameWidth = frameOpencvDnn.shape[1]
    # Preprocess the image loaded into dnn and convert it to blob format framOpencvdnn: Import image; 1.0: Scale is 1
    # without scaling; (300,300):Neural Network size() [104,117,123]:mean subtraction value(that is, the R, G,
    # and B channels of the picture should be True: Opencv usually assumes that the picture channel is BGR,
    # and the mean value is RGB. False: No cropping operation, no clipping operation by default Reference: [Deep
    # learning: How OpenCV’s blobFromImage works](
    # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Load network input data(picture)
    net.setInput(blob)

    # Assign the result of the forward transmission of the neural network to the dete
    detections = net.forward()

    # Storing the coordinates of the face image
    faceBoxes = []

    # Get the confidence of the facial image, coordinates (x1,y1) (x2, y2), frame the face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # Original image, point coordinates, point coordinates, RGB color corresponding to the line,
            # line thickness, line type Note: The coordinates in the rectangle are diagonal coordinates, which may be
            # upper left-lower right, or upper right-lower left, here is upper right-bottom left. :[Drawing
            # Functions](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#cv2.rectangle)
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


# argparse is a built-in module for command item options and parameter parsing in Python. It is actually displaying
# help information and specifying parameter input methods in the terminal, parsing the parameters of the delivery
# terminal. Detailed use see: [Python super easy to use standard library argparse] (
# https://medium.com/@dboyliao/python-super easy to use standard library - argparse-4eab2e9dcc69)
parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

# Define the model used for facial recognition and its related configuration files
# :proto + pbtxt-- [TensorFlow simple pb (pbtxt) file read and write] (https://www.jianshu.com/p/3de6ffc490a9)
# # -- [A text to understand Protocol Buffer] (https://zhuanlan.zhihu.com/p/36554982)
# # proto + pb -- [How many model formats does TensorFlow have? ](https://cloud.tencent.com/developer/article/1009979)
# # :[Files of TensorFlow](https://blog.csdn.net/lrglgy/article/details/89484078)
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age list [0-2], [4-6],...
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Gender List
genderList = ['Male', 'Female']

# Load network
# Loading face recognition, age discrimination, gender discrimination network
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open a video file or an image file or a camera stream
# When the terminal input parameter is not empty, open the corresponding picture. If it is empty, open the camera.
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20


while cv2.waitKey(1) < 0:
    # read video frame sequence
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # Pass the face-recognized picture to frameFace, the face-related coordinates are passed to faceBoxes
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    # traverse face coordinates
    for faceBox in faceBoxes:
        # Select a minimum area containing the face image and assign it to face
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                    : min(faceBox[2] + padding, frame.shape[1] - 1)]

# Pre-process the acquired facial image: scaling, neural network size, mean subtraction value, whether to exchange R
        # channel and B channel
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        # Return the coordinates corresponding to the maximum value of the probability, and select the corresponding
        # gender in the gender list to assign to the gender.
        gender = genderList[genderPreds[0].argmax()]
        # Print the gender output
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
