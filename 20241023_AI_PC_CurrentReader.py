# this scripts opens a USB camera and displays the video stream
# it also uses AI to find a 4digit (7segmented)  number in the video stream
# the number is then displayed on the screen
# the number is also used to create a graph which is updated every second
# it uses a gui to display the video stream and the graph

# thinter is used for the gui
# cv2 is used for the video stream
# numpy is used for the AI
# time is used for the delay

import tkinter as tk
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageTk
from matplotlib import style
from matplotlib import rcParams
from matplotlib import ticker

#setup the gui
root = tk.Tk()
root.title("Number Detection")
root.geometry("800x600")
root.resizable(0,0)

#setup the video stream
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

#setup the AI
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

#function to show the video stream in the gui  
#original frame and processed frame are displayed
def show_frame():
    _, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
    #cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #img = Image.fromarray(cv2image)
    #imgtk = ImageTk.PhotoImage(image=img)
    #lmain.imgtk = imgtk
    #lmain.configure(image=imgtk)
    lmain.after(5, show_frame)
    boundBoxes = find_text(frame)
    #show rectangles around the text
    #for (startX, startY, endX, endY) in boundBoxes:
    #    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
    #cv2.imshow("Text Detection", frame)

def find_text(frame):
    # EAST model requires the width and height of the image to be multiple of 32
    #orig = frame.copy()
    (H, W) = frame.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    frame = cv2.resize(frame, (newW, newH))
    (H, W) = frame.shape[:2]
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    # add layers of network
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # create a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward(layerNames)
    scores = output[0]
    geometry = output[1]
    # get the bounding boxes
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if score is lower than the confidence threshold, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    #TODO verify if this is needed: boxes = non_max_suppression(np.array(rects), probs=confidences)
    # scale image back to original size
    for (startX, startY, endX, endY) in rects:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
    # show rectangles around the text
    for (startX, startY, endX, endY) in rects:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
    cv2.imshow("Text Detection", frame)
    return rects

if __name__ == "__main__":
    lmain = tk.Label(root)
    lmain.pack()
    show_frame()
    root.mainloop()
