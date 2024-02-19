
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

from xarmlib import version
from xarmlib.wrapper import XArmAPI

import pygame
#######################################################

class RoboticArmAssembly:
    def __init__(self):
        self.arm = XArmAPI('192.168.1.207', baud_checkset=False)
        self.variables = {}
        self.params = {
            'speed': 100,
            'acc': 2000,
            'angle_speed': 20,
            'angle_acc': 500,
            'events': {},
            'variables': self.variables,
            'callback_in_thread': True,
            'quit': False
        }
        #self.setup_callbacks()

    def cameraCheck(self):
        path = 'C:/Users/jongh/projects/llm-roboticarm/vision_data/check.pt'
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

    def objectPlace(self, objectType):
        path = 'C:/Users/jongh/projects/llm-roboticarm/vision_data/{}.pt'.format(objectType)
        # model = None
        print(objectType)
        print(path)
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=False)

        cap = cv2.VideoCapture(0)
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
            
    def distinguish_orientation(self, image):
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

    def pprint(self, *args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                    ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    # Register error/warn changed callback
    def error_warn_change_callback(self, data):
        if data and data['error_code'] != 0:
            self.params['quit'] = True
            self.pprint('err={}, quit'.format(data['error_code']))
            self.arm.release_error_warn_changed_callback(self.error_warn_change_callback)

    # Register state changed callback
    def state_changed_callback(self, data):
        if data and data['state'] == 4:
            if self.arm.version_number[0] > 1 or (self.arm.version_number[0] == 1 and self.arm.version_number[1] > 1):
                self.params['quit'] = True
                self.pprint('state=4, quit')
                self.arm.release_state_changed_callback(self.state_changed_callback)          

    # Register connect changed callback
    def connect_changed_callback(self, data):
        if data and not data['connected']:
            self.params['quit'] = True
            self.pprint('disconnect, connected={}, reported={}, quit'.format(data['connected'], data['reported']))
            self.arm.release_connect_changed_callback(self.error_warn_change_callback)

    def display_video(self, index, name, x, y):
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

    def housing_movement(self):
        if self.x2h - self.x1h <= self.y2h - self.y1h:  # THIS IS FOR HOUSING BEFORE ELSE
            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(600, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xhouse - 15, self.yhouse, 30.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xhouse - 15, self.yhouse, 7, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(10, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[self.xhouse, self.yhouse - 13, 30, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)  
                code = self.arm.set_position(*[94, -280, 65, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-85, -339, 65, 180.0, -90.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-85, -339, 32.4, 180.0, -90.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(850, wait=True, speed=800, auto_enable=True)
                code = self.arm.set_position(*[-80, -337.3, 35, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
        else:
            if self.arm.error_code == 0 and not self.params['quit']:  # THIS PART OF CODE IS ALSO FOR HOUSING AFTER ELSE
                code = self.arm.set_gripper_position(600, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xhouse, self.yhouse, 30.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xhouse, self.yhouse, 10, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(100, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[self.xhouse, self.yhouse - 13, 30, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 65, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-85, -339, 65, 180.0, -90.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-85, -339, 32.4, 180.0, -90.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(850, wait=True, speed=800, auto_enable=True)
                code = self.arm.set_position(*[-80, -337.3, 35, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                if code != 0:
                    self.params['quit'] = True
                    self.pprint('set_position, code={}'.format(code))

    def wedge_movement(self):
        if self.x2w - self.x1w <= self.y2w - self.y1w:
            temp = 0  # For when the ridges are up and the ramp is pointing to the top of the camera view BEFORE ELSE
            if self.arm.error_code == 0 and not self.params['quit'] and temp == 0:
                code = self.arm.set_position(*[-80, -337.3, 60, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_gripper_position(250, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xwedge + 8.5, self.ywedge - 17, 30.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xwedge + 8.5, self.ywedge - 17, 6.5, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-74.1, -335.7, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-74.1, -335.7, 56.4, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[-81.8, -330.8, 74.5, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
        else:
        # For when the ridges are up and the ramp is pointing to the side of the camera view AFTER ELSE
            code = self.arm.set_position(*[-80, -337.3, 60, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                    mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_gripper_position(250, wait=True, speed=800, auto_enable=True)

            code = self.arm.set_position(*[self.xwedge-8, self.ywedge, 30.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
            code = self.arm.set_position(*[self.xwedge-8, self.ywedge, 6.5, 180.0, 0.0, 90.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

            code = self.arm.set_gripper_position(1, wait=True, speed=800, auto_enable=True)  # This is for the spring
            code = self.arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                radius=-1.0, wait=True)
            code = self.arm.set_position(*[-74.1, -335.7, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                radius=-1.0, wait=True)
            code = self.arm.set_position(*[-74.1, -335.7, 56.4, 180.0, -90.0, 0.0], speed=50,
                                mvacc=self.params['acc'],
                                radius=-1.0, wait=True)

            code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)
            code = self.arm.set_position(*[-81.8, -330.8, 74.5, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                mvacc=self.params['acc'],
                                radius=-1.0, wait=True)

            code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                radius=-1.0, wait=True)

    def spring_movement(self):
        if self.x2s - self.x1s <= self.y2s - self.y1s:
            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xspring + 10, self.yspring - 17, 30.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xspring + 10, self.yspring - 17, 5, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-74.5, -333.8, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-74.5, -333.8, 58.2, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

        else:

            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xspring + 10, self.yspring - 10, 30.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xspring + 10, self.yspring - 10, 5, 180.0, 0.0, 90.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                code = self.arm.set_position(*[self.xspring + 15, self.yspring - 9, 15, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-74.5, -333.8, 95, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-74.5, -333.8, 58.2, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                
    def cap_movement(self):
            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xcap-5, self.ycap - 20, 30.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xcap-5, self.ycap - 20,4.5, 180.0, 0.0, 0.0], speed=75,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[240, -340, 65, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                #code = arm.set_position(*[94, -280, 95, 180.0, 0.0, 0.0], speed=params['speed'],
                                        #mvacc=params['acc'],
                                        #radius=-1.0, wait=True)
                code = self.arm.set_position(*[-65.5, -334.9, 95, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-65.5, -334.9, 48.2, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[-72.2, -332.6, 95, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                if code != 0:
                    self.params['quit'] = True
                    self.pprint('set_position, code={}'.format(code))

            self.params['quit'] = True
            self.pprint('set_position, code={}'.format(code))

            if code != 0:
                self.params['quit'] = True
                self.pprint('set_position, code={}'.format(code))

    def cleanup(self):
        # Stop any background threads or asynchronous operations
        self.params['quit'] = True  # Assuming this flag stops threads started by this application

        # Release the robotic arm connection
        if self.arm.is_connected():
            self.arm.disconnect()

        # Close any OpenCV camera captures
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

        # Close any Tkinter windows if they're open
        for window in tk.Tk().tk_windowManager():
            window.destroy()

        # Stop any pygame audio playback
        pygame.mixer.music.stop()
        pygame.quit()

        # Additional cleanup actions as needed
        print("Cleanup complete. Resources have been released.")
        
    def robotic_assembly(self):
        self.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))        
        self.arm.register_error_warn_changed_callback(self.error_warn_change_callback)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(1)

        self.arm.register_state_changed_callback(self.state_changed_callback)

        # Register counter value changed callback
        if hasattr(self.arm, 'register_count_changed_callback'):
            def count_changed_callback(data):
                if not self.params['quit']:
                    self.pprint('counter val: {}'.format(data['count']))
            self.arm.register_count_changed_callback(count_changed_callback)

        self.arm.register_connect_changed_callback(self.connect_changed_callback)

        if not self.params['quit']:
            self.params['speed'] = 150
        if not self.params['quit']:
            self.params['acc'] = 10000
        for i in range(int(10)):
            if self.params['quit']:
                break

        movement = 1
        if movement == 1:
            self.x1h, self.y1h, self.x2h, self.y2h, a, b, c = self.objectPlace('housing')
            self.x1c, self.y1c, self.x2c, self.y2c, a, b, c= self.objectPlace('cap')
            self.x1s, self.y1s, self.x2s, self.y2s, a, b, c = self.objectPlace('spring')
            self.x1w, self.y1w, self.x2w, self.y2w, mask, mask1, mask2 = self.objectPlace('wedge')

            print("Top left: ({},{}) Bottom right: ({},{})".format(self.x1w, self.y1w, self.x2w, self.y2w))

            print(self.x1w, self.y1w, self.x2w, self.y2w)

            wedgeTest = self.distinguish_orientation(mask)
            wedgeTest1 = self.distinguish_orientation(mask1)
            wedgeTest2 = self.distinguish_orientation(mask2)
            print(wedgeTest1)
            print(wedgeTest2)
            if wedgeTest <= 0.2 and wedgeTest >= 0.12:
                print("Ridged")
            elif wedgeTest >= 0.2:
                print("Smooth")

            # x=(10/13)(ync)+250
            # y=(5/7)(xnc)-363
            self.xhouse = int(self.y1h + self.y2h)
            self.xhouse = self.xhouse * 5 / 7
            self.xhouse = self.xhouse / 2 + 250

            self.yhouse = int(self.x1h + self.x2h)
            self.yhouse = self.yhouse * 5 / 7
            self.yhouse = self.yhouse / 2 - 363

            self.xwedge = int(self.y1w + self.y2w)
            self.xwedge = self.xwedge * 5 / 7
            self.xwedge = self.xwedge / 2 + 250

            self.ywedge = int(self.x1w + self.x2w + 480)
            self.ywedge = self.ywedge * 5 / 7
            self.ywedge = self.ywedge / 2 - 363

            self.xspring = int(self.y1s + self.y2s)
            self.xspring = self.xspring * 5 / 7
            self.xspring = self.xspring / 2 + 250

            self.yspring = int(self.x1s + self.x2s + 880)
            self.yspring = self.yspring * 5 / 7
            self.yspring = self.yspring / 2 - 363

            self.xcap = int(self.y1c + self.y2c)
            self.xcap = self.xcap * 5 / 7
            self.xcap = self.xcap / 2 + 250

            self.ycap = int(self.x1c + self.x2c + 880)
            self.ycap = self.ycap * 5 / 7
            self.ycap = self.ycap / 2 - 363

            self.housing_movement()
            self.wedge_movement()

            # cameraCheck()
            # blank

            self.spring_movement()

            self.cap_movement()
            self.cap_movement()
            self.cap_movement()

        # release all event
        if hasattr(self.arm, 'release_count_changed_callback'):
            self.arm.release_count_changed_callback(count_changed_callback)
        self.arm.release_error_warn_changed_callback(self.state_changed_callback)
        self.arm.release_state_changed_callback(self.state_changed_callback)
        self.arm.release_connect_changed_callback(self.error_warn_change_callback)

if __name__ == "__main__":
    assembly = RoboticArmAssembly()
    try:
        assembly.robotic_assembly()
    finally:
        assembly.cleanup()