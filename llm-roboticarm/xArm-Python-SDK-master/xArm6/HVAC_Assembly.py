
import os
import threading
import sys

import time

import traceback
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch

import tkinter as tk
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from xarmlib.tools import utils
except:
    pass
from xarmlib import version
from xarmlib.wrapper import XArmAPI
import pygame
#######################################################
"""
Just for test example
"""
errorDict = {'housing': [], 'wedge': [], 'spring': [], 'cap': []}

def cameraCheck():
    path = 'C:/Users/jongh/projects/xArm-Python-SDK-master/example/wrapper/vision_data/check.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=False)
    cap = cv2.VideoCapture(4)
    while True:
        ret, frame = cap.read()
        #frame = frame[121:264, 223:393]
        results = model(frame)
        coords_plus = results.pandas().xyxy[0]
        if coords_plus.empty:
            print("No object found. Successful wedge placement")

        if not coords_plus.empty:
            for index, row in coords_plus.iterrows():
                name = row['name']

                if name == 'wedge':
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("CHECK", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        return

def objectPlace(objectType):
    path = 'C:/Users/jongh/projects/xArm-Python-SDK-master/example/wrapper/vision_data/{}.pt'.format(objectType)
    # model = None
    model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=False)

    cap = cv2.VideoCapture(1)
    temp = 0
    temp2 = 0
    if objectType=='housing':
        objectType = 'housing-flat'
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        frame1 = frame
        if objectType == 'housing-flat':
            frame = frame[0:480, 0:240] # housing
        elif objectType == 'wedge':
            frame = frame[0:480, 240:440]
        elif objectType == 'spring':
            frame = frame[0:480, 440:640]
        else:
            frame = frame[0:480, 440:640]
        mask = frame
        mask1 = frame
        mask2 = frame
        results = model(frame)
        coords_plus = results.pandas().xyxy[0]
        if coords_plus.empty:
            print("No object found")
            exit()
        if not coords_plus.empty and temp == 0:
            for index, row in coords_plus.iterrows():
                name = row['name']

                if name == objectType:
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    temp = 1
                    masky = y2 + y1 - 2
                    masky = masky / 2
                    masky = int(masky)
                    # if abs((x2 -x1 ) -(y2 -y1)) <= 10 and objectType == 'housing-flat':
                    #     temp = 2
        if temp == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            mask = mask[y1 - 2:y2, x1 - 3:x2 + 3]
            mask1 = mask1[y1 - 2:masky, x1 - 3:x2 + 3]
            mask2 = mask2[masky:y2, x1 - 3:x2 + 3]
        elif temp == 2:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame, (x1, y2), (x2, y1), (0, 0, 255), 2)
            cv2.line(frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame1, (x1, y2), (x2, y1), (0, 0, 255), 2)
        if objectType == 'wedge':
            cv2.imshow("mask", mask)
            cv2.imshow("mask1", mask1)
            cv2.imshow("mask2", mask2)
        cv2.imshow("FRAME", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        if temp2 == 0 and temp == 2:
            # errorDict[objectType] += [[x1, y1, x2, y2]]
            # print(errorDict)
            x = (x1 +x2 ) /2
            y = (y1 +y2 ) /2
            pygame.init()
            housing_error_file = r'C:\Users\ade5221\Downloads\housing_error_new_trim.wav'
            pygame.mixer.music.load(housing_error_file)
            pygame.mixer.music.play()

            top = tk.Tk()
            top.title('Error')
            labelString = "{} error centered at (" + str(x) + ',' + str(y) + ')'.format(objectType)
            label_text = tk.Label(top, text=labelString)
            frame1_forPIL = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            newImage = Image.fromarray(frame1_forPIL)
            add = ImageTk.PhotoImage(image=newImage)
            frame1_label = tk.Label(top, image=add)
            label_text.pack()
            frame1_label.pack()
            top.mainloop()
            # Code here will only execute after the GUI window is closed
            exit()
        elif temp2 == 0 and temp != 2:
            return x1, y1, x2, y2, mask, mask1, mask2
def distinguish_orientation(image):
    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Calculate edge orientation
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    edge_orientation = np.arctan2(sobely, sobelx)

    # Calculate statistics for edge orientations
    orientation_mean = np.mean(edge_orientation)
    orientation_std = np.std(edge_orientation)

    return orientation_mean




def pprint(*args, **kwargs):
    try:
        stack_tuple = traceback.extract_stack(limit=2)[0]
        print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                   ' '.join(map(str, args))))
    except:
        print(*args, **kwargs)


pprint('xArm-Python-SDK Version:{}'.format(version.__version__))

arm = XArmAPI('192.168.1.207', baud_checkset=False)
arm.clean_warn()
arm.clean_error()
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)

variables = {}
params = {'speed': 100, 'acc': 2000, 'angle_speed': 20, 'angle_acc': 500, 'events': {}, 'variables': variables,
          'callback_in_thread': True, 'quit': False}


# Register error/warn changed callback
def error_warn_change_callback(data):
    if data and data['error_code'] != 0:
        params['quit'] = True
        pprint('err={}, quit'.format(data['error_code']))
        arm.release_error_warn_changed_callback(error_warn_change_callback)


arm.register_error_warn_changed_callback(error_warn_change_callback)


# Register state changed callback
def state_changed_callback(data):
    if data and data['state'] == 4:
        if arm.version_number[0] > 1 or (arm.version_number[0] == 1 and arm.version_number[1] > 1):
            params['quit'] = True
            pprint('state=4, quit')
            arm.release_state_changed_callback(state_changed_callback)


arm.register_state_changed_callback(state_changed_callback)

# Register counter value changed callback
if hasattr(arm, 'register_count_changed_callback'):
    def count_changed_callback(data):
        if not params['quit']:
            pprint('counter val: {}'.format(data['count']))


    arm.register_count_changed_callback(count_changed_callback)


# Register connect changed callback
def connect_changed_callback(data):
    if data and not data['connected']:
        params['quit'] = True
        pprint('disconnect, connected={}, reported={}, quit'.format(data['connected'], data['reported']))
        arm.release_connect_changed_callback(error_warn_change_callback)


arm.register_connect_changed_callback(connect_changed_callback)


def display_video(index, name, x, y):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cv2.namedWindow(name)
    cv2.moveWindow(name, x, y)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows(name)

if not params['quit']:
    params['speed'] = 150
if not params['quit']:
    params['acc'] = 10000
for i in range(int(10)):
    if params['quit']:
        break
    
def housing_movement():
    if x2h - x1h <= y2h - y1h:  # THIS IS FOR HOUSING BEFORE ELSE
        if arm.error_code == 0 and not params['quit']:
            code = arm.set_gripper_position(600, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[xhouse - 15, yhouse, 30.5, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[xhouse - 15, yhouse, 7, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = arm.set_gripper_position(10, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = arm.set_position(*[xhouse, yhouse - 13, 30, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)  
            code = arm.set_position(*[94, -280, 65, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-85, -339, 65, 180.0, -90.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-85, -339, 32.4, 180.0, -90.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_gripper_position(850, wait=True, speed=800, auto_enable=True)
            code = arm.set_position(*[-80, -337.3, 35, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
    else:
        if arm.error_code == 0 and not params['quit']:  # THIS PART OF CODE IS ALSO FOR HOUSING AFTER ELSE
            code = arm.set_gripper_position(600, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[xhouse, yhouse, 30.5, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[xhouse, yhouse, 10, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = arm.set_gripper_position(100, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = arm.set_position(*[xhouse, yhouse - 13, 30, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-85, -339, 65, 180.0, -90.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-85, -339, 32.4, 180.0, -90.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_gripper_position(850, wait=True, speed=800, auto_enable=True)
            code = arm.set_position(*[-80, -337.3, 35, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            if code != 0:
                params['quit'] = True
                pprint('set_position, code={}'.format(code))
def wedge_movement():
    if x2w - x1w <= y2w - y1w:
        temp = 0  # For when the ridges are up and the ramp is pointing to the top of the camera view BEFORE ELSE
        if arm.error_code == 0 and not params['quit'] and temp == 0:
            code = arm.set_position(*[-80, -337.3, 60, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_gripper_position(250, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[xwedge + 8.5, ywedge - 17, 30.5, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[xwedge + 8.5, ywedge - 17, 6.5, 180.0, 0.0, 0.0], speed=75, mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = arm.set_gripper_position(1, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-74.1, -335.7, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-74.1, -335.7, 56.4, 180.0, -90.0, 0.0], speed=50,
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[-81.8, -330.8, 74.5, 180.0, -90.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
    else:
       # For when the ridges are up and the ramp is pointing to the side of the camera view AFTER ELSE
        code = arm.set_position(*[-80, -337.3, 60, 180.0, 0.0, 0.0], speed=params['speed'],
                                mvacc=params['acc'],
                                radius=-1.0, wait=True)
        code = arm.set_gripper_position(250, wait=True, speed=800, auto_enable=True)

        code = arm.set_position(*[xwedge-8, ywedge, 30.5, 180.0, 0.0, 90.0], speed=params['speed'],
                                     mvacc=params['acc'],  # For wedge use y-10 instead of y
                                     radius=-1.0, wait=True)
        code = arm.set_position(*[xwedge-8, ywedge, 6.5, 180.0, 0.0, 90.0], speed=75, mvacc=params['acc'],
                                     radius=-1.0, wait=True)

        code = arm.set_gripper_position(1, wait=True, speed=800, auto_enable=True)  # This is for the spring
        code = arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                            radius=-1.0, wait=True)
        code = arm.set_position(*[-74.1, -335.7, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                            radius=-1.0, wait=True)
        code = arm.set_position(*[-74.1, -335.7, 56.4, 180.0, -90.0, 0.0], speed=50,
                            mvacc=params['acc'],
                            radius=-1.0, wait=True)

        code = arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)
        code = arm.set_position(*[-81.8, -330.8, 74.5, 180.0, -90.0, 0.0], speed=params['speed'],
                            mvacc=params['acc'],
                            radius=-1.0, wait=True)

        code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                            radius=-1.0, wait=True)




def spring_movement():
    if x2s - x1s <= y2s - y1s:
        if arm.error_code == 0 and not params['quit']:
            code = arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[xspring + 10, yspring - 17, 30.5, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[xspring + 10, yspring - 17, 5, 180.0, 0.0, 0.0], speed=75, mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = arm.set_gripper_position(1, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-74.5, -333.8, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-74.5, -333.8, 58.2, 180.0, -90.0, 0.0], speed=50,
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)

    else:

        if arm.error_code == 0 and not params['quit']:
            code = arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[xspring + 10, yspring - 10, 30.5, 180.0, 0.0, 90.0], speed=params['speed'],
                                    mvacc=params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[xspring + 10, yspring - 10, 5, 180.0, 0.0, 90.0], speed=75, mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = arm.set_gripper_position(1, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            code = arm.set_position(*[xspring + 15, yspring - 9, 15, 180.0, 0.0, 0.0], speed=75, mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-74.5, -333.8, 95, 180.0, -90.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-74.5, -333.8, 58.2, 180.0, -90.0, 0.0], speed=50,
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'], mvacc=params['acc'],
                                    radius=-1.0, wait=True)
def cap_movement():

        if arm.error_code == 0 and not params['quit']:
            code = arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[xcap-5, ycap - 20, 30.5, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[xcap-5, ycap - 20,4.5, 180.0, 0.0, 0.0], speed=75,
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = arm.set_gripper_position(1, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            #code = arm.set_position(*[94, -280, 95, 180.0, 0.0, 0.0], speed=params['speed'],
                                    #mvacc=params['acc'],
                                    #radius=-1.0, wait=True)
            code = arm.set_position(*[-65.5, -334.9, 95, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            code = arm.set_position(*[-65.5, -334.9, 48.2, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)

            code = arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

            code = arm.set_position(*[-72.2, -332.6, 95, 180.0, 0.0, 0.0], speed=params['speed'],
                                    mvacc=params['acc'],
                                    radius=-1.0, wait=True)
            if code != 0:
                params['quit'] = True
                pprint('set_position, code={}'.format(code))

        params['quit'] = True
        pprint('set_position, code={}'.format(code))

        if code != 0:
            params['quit'] = True
            pprint('set_position, code={}'.format(code))

movement = 1
if movement == 1:
     x1h, y1h, x2h, y2h, a, b, c = objectPlace('housing')
     x1c, y1c, x2c, y2c, a, b, c= objectPlace('cap')
     x1s, y1s, x2s, y2s, a, b, c = objectPlace('spring')
     x1w, y1w, x2w, y2w, mask, mask1, mask2 = objectPlace('wedge')

     print("Top left: ({},{}) Bottom right: ({},{})".format(x1w, y1w, x2w, y2w))

     print(x1w, y1w, x2w, y2w)

     wedgeTest = distinguish_orientation(mask)
     wedgeTest1 = distinguish_orientation(mask1)
     wedgeTest2 = distinguish_orientation(mask2)
     print(wedgeTest1)
     print(wedgeTest2)
     if wedgeTest <= 0.2 and wedgeTest >= 0.12:
         print("Ridged")
     elif wedgeTest >= 0.2:
         print("Smooth")

     # x=(10/13)(ync)+250
     # y=(5/7)(xnc)-363
     xhouse = int(y1h + y2h)
     xhouse = xhouse * 5 / 7
     xhouse = xhouse / 2 + 250

     yhouse = int(x1h + x2h)
     yhouse = yhouse * 5 / 7
     yhouse = yhouse / 2 - 363

     xwedge = int(y1w + y2w)
     xwedge = xwedge * 5 / 7
     xwedge = xwedge / 2 + 250

     ywedge = int(x1w + x2w + 480)
     ywedge = ywedge * 5 / 7
     ywedge = ywedge / 2 - 363

     xspring = int(y1s + y2s)
     xspring = xspring * 5 / 7
     xspring = xspring / 2 + 250

     yspring = int(x1s + x2s + 880)
     yspring = yspring * 5 / 7
     yspring = yspring / 2 - 363

     xcap = int(y1c + y2c)
     xcap = xcap * 5 / 7
     xcap = xcap / 2 + 250

     ycap = int(x1c + x2c + 880)
     ycap = ycap * 5 / 7
     ycap = ycap / 2 - 363
     housing_movement()


     wedge_movement()

     # cameraCheck()
     # blank

     spring_movement()

     cap_movement()
     cap_movement()
     cap_movement()


# release all event
if hasattr(arm, 'release_count_changed_callback'):
    arm.release_count_changed_callback(count_changed_callback)
arm.release_error_warn_changed_callback(state_changed_callback)
arm.release_state_changed_callback(state_changed_callback)
arm.release_connect_changed_callback(error_warn_change_callback)

