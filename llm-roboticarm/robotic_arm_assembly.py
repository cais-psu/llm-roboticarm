
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
import math
import pygame
import openai_agent
from pyzbar.pyzbar import decode

#######################################################

class RoboticArmAssembly:
    def __init__(self):
        self.arm = XArmAPI('192.168.1.240', baud_checkset=False)
        self.variables = {}
        self.params = {
            'speed': 180,
            'acc': 10000,
            'angle_speed': 20,
            'angle_acc': 500,
            'events': {},
            'variables': self.variables,
            'callback_in_thread': True,
            'quit': False
        }
        self.step_already_done = None

    def detect_qr_code_from_camera(self):
        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            decoded_objects = decode(frame)
            for obj in decoded_objects:
                (self.x, self.y, self.w, self.h) = obj.rect
                cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
            cv2.imshow("QR Code Detection", frame)
            time.sleep(1)
            break
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
        return (self.x+self.w//2,self.y+self.w//2)

    def cameraCheck(self, assembly_step):
        path = 'llm-roboticarm/vision_data/check.pt'
        
        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
        cap = cv2.VideoCapture(1)
        temp=1
        while True and temp<10:
            ret, frame = cap.read()
            #frame = frame[121:264, 223:393]
            results = model(frame)
            coords_plus = results.pandas().xyxy[0]
                        
            if coords_plus.empty:
                cv2.imshow("CHECK", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                temp+=1
            
            if not coords_plus.empty:
                for index, row in coords_plus.iterrows():
                    assembly_step = row['name']

                    if assembly_step == 'wedge':
                        x1 = int(row['xmin'])
                        y1 = int(row['ymin'])
                        x2 = int(row['xmax'])
                        y2 = int(row['ymax'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                cv2.imshow("CHECK", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                print("Object found.")
                time.sleep(1)
                cap.release()
                cv2.destroyAllWindows()
                exit()
                
        print('No object found. Wedge placement successful.')    
        return

    def objectPlace(self, objectType):
        try:
            path = 'llm-roboticarm/vision_data/{}.pt'.format(objectType)
            # model = None
            print(objectType)
            print(path)
            
            model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
            cap = cv2.VideoCapture(1)
            #cap = cv2.VideoCapture(0)
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
                    print(f"No object found: {objectType}")
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
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Clean up: Release the capture object and close any OpenCV windows
            cap.release()
            cv2.destroyAllWindows()
                  
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

                code = self.arm.set_position(*[self.xH, self.yH, 190.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xH, self.yH, 178.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(175, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[self.xQRarm+(self.yhouse-self.yQRPix)*21/32, self.yQRarm+(self.xhouse-self.xQRPix)*21/32, 208.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 243.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 243.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[94, -280, 243.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-85, -510.2, 70, 180.0, -90.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-85, -510.2, 40, 180.0, -90.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(850, wait=True, speed=800, auto_enable=True)
                code = self.arm.set_position(*[-85, -510.2, 100, 180.0, -90.0, 90.0], speed=self.params['speed'],mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-80, -337.3, 213.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
        else:
            if self.arm.error_code == 0 and not self.params['quit']:  # THIS PART OF CODE IS ALSO FOR HOUSING AFTER ELSE
                code = self.arm.set_gripper_position(600, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xH, self.yH, 30.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xH,self.yH, 10, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(100, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[self.xhouse, self.yhouse - 13, 165, 180.0, 0.0, 90.0], speed=self.params['speed'],
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
                code = self.arm.set_position(*[-80, -337.3, 250, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_gripper_position(250, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xW,self.yW, 195.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xW,self.yW , 176, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                code = self.arm.set_position(*[self.xW, -291, 225, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[94, -280, 270, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-243.5, -335.6, 100, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-243.5, -335.6, 54, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[-81.8, -330.8, 250, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_position(*[240, -340, 240, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
        else:
        # For when the ridges are up and the ramp is pointing to the side of the camera view AFTER ELSE
            code = self.arm.set_position(*[-80, -337.3, 60, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                    mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_gripper_position(250, wait=True, speed=800, auto_enable=True)

            code = self.arm.set_position(*[self.xW,self.yW, 30.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
            code = self.arm.set_position(*[self.xW,self.yW, 6.5, 180.0, 0.0, 90.0], speed=75, mvacc=self.params['acc'],
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

                code = self.arm.set_position(*[self.xS, self.yS, 195.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xS, self.yS, 175, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                code = self.arm.set_position(*[self.xS, -291, 225, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[94, -280, 270, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-244.5, -335.7, 100, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-244.5, -335.7, 54, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[-81.8, -330.8, 250, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

        else:
            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xS, self.yS, 30.5, 180.0, 0.0, 90.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xS, self.yS, 5, 180.0, 0.0, 90.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                code = self.arm.set_position(*[self.xS, self.yS, 15, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
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

            code = self.arm.set_position(*[self.xC, self.yC, 195.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                    mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = self.arm.set_position(*[self.xC, self.yC, 177, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)

            # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            code = self.arm.set_position(*[289.4, -27.6, 237, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = self.arm.set_position(*[289.4, -322, 237, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_position(*[-71.5, -335.8, 237, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_position(*[-71.5, -335.8, 217, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)
            #
            # code = self.arm.set_position(*[-81.8, -330.8, 250, 180.0, -90.0, 0.0], speed=self.params['speed'],
            #                         mvacc=self.params['acc'],
            #                         radius=-1.0, wait=True)
            if code != 0:
                self.params['quit'] = True
                self.pprint('set_position, code={}'.format(code))

        self.params['quit'] = True
        self.pprint('set_position, code={}'.format(code))

        if code != 0:
            self.params['quit'] = True
            self.pprint('set_position, code={}'.format(code))

    def perform_housing_step(self):
        try:
            self.count_and_display_objects()
        except:
            message = f"Error: the housing object is overlapping"
            return message            
        
        try:
            self.x1h, self.y1h, self.x2h, self.y2h, a, b, c = self.objectPlace('housing')
            
            self.xhouse = int(self.x1h + self.x2h)
            self.xhouse = self.xhouse /2

            self.yhouse = int(self.y1h + self.y2h)
            self.yhouse = self.yhouse /2

            self.xH = self.xQRarm + (self.yhouse - self.yQRPix)*0.65
            self.yH = self.yQRarm + (self.xhouse - self.xQRPix) * 0.64            
        except:
            message = f"Error: the housing object was not detected"
            return message

        try:
            self.housing_movement()
        except Exception as e:
            message = f"Error {e}: there was an error during the housing movement"
            return message
                    
        self.step_already_done = "housing"
        message = "Housing step completed successfully."
        return message          

        
    def perform_wedge_step(self):
        try:
            self.x1w, self.y1w, self.x2w, self.y2w, mask, mask1, mask2 = self.objectPlace('wedge')

            self.xwedge = int(self.x1w + self.x2w+480)
            self.xwedge = self.xwedge /2

            self.ywedge = int(self.y1w + self.y2w)
            self.ywedge = self.ywedge /2

            self.xW = self.xQRarm + (self.ywedge - self.yQRPix) * 0.65
            self.yW = self.yQRarm + (self.xwedge - self.xQRPix) * 0.64
        except:
            return f"Error: during wedge object placement detection"

        try:
            self.wedge_movement()
        except Exception as e:
            return f"Error {e} during wedge movement"
                        
        #try:
            #self.cameraCheck("wedge")
        #except:
            #### temporary for test ####
            #self.step_already_done = "wedge"
            ############################
            #return f"Error: wedge object placement is not done correctly."
    
        self.step_already_done = "wedge"
        
        return "Wedging step completed successfully."
                
    def perform_spring_step(self):
        try:
            self.x1s, self.y1s, self.x2s, self.y2s, a, b, c = self.objectPlace('spring')

            self.xspring = int(self.x1s + self.x2s+880)
            self.xspring = self.xspring / 2
            # # #
            self.yspring = int(self.y1s + self.y2s)
            self.yspring = self.yspring / 2
            #
            self.xS = self.xQRarm + (self.yspring - self.yQRPix) * 0.65
            self.yS = self.yQRarm + (self.xspring - self.xQRPix) * 0.64
        except:
            message = f"Error: the spring object was not detected"
            return message

        try:
            self.spring_movement()
        except Exception as e:
            return f"Error {e} during spring movement"
                        
        self.step_already_done = "spring"
        return "Spring step completed successfully."

    def perform_cap_step(self):
        try:
            self.x1c, self.y1c, self.x2c, self.y2c, a, b, c = self.objectPlace('cap')

            self.xcap = int(self.x1c + self.x2c+880)
            self.xcap = self.xcap / 2

            self.ycap = int(self.y1c + self.y2c)
            self.ycap = self.ycap / 2

            self.xC = self.xQRarm + (self.ycap - self.yQRPix) * 0.65
            self.yC = self.yQRarm + (self.xcap - self.xQRPix) * 0.64
        except:
            return f"Error: {e} during cap object placement detection"

        try:
            self.cap_movement()
        except Exception as e:
            return f"Error {e} during cap movement"
                        
        self.step_already_done = "completed"
        return "Cap step completed successfully."
        
    def resume_assembly_from_last_step(self, step_already_done):
        # Define the order of assembly steps
        assembly_steps = ["housing", "wedge", "spring", "cap"]
        #assembly_steps = ["housing", "wedge", "spring"]
        last_completed_index = assembly_steps.index(step_already_done) if step_already_done in assembly_steps else -1
        
        for step in assembly_steps[last_completed_index + 1:]:
            message = getattr(self, f"perform_{step}_step")()
            if 'error' in message.lower():
                return self.step_already_done, message
        
        self.step_already_done = "completed"
        return self.step_already_done, "All steps for the assembly are successfully completed."

    def start_robotic_assembly(self):

        self.xQRPix,self.yQRPix = self.detect_qr_code_from_camera()
        self.xQRarm=293
        self.yQRarm=-130

        for step in ["housing", "wedge", "spring", "cap"]:
        #for step in ["housing", "wedge", "spring"]:
            message = getattr(self, f"perform_{step}_step")()
            if 'error' in message.lower():
                return self.step_already_done, message
        
        return self.step_already_done, message
    
    def find_available_cameras(self):
        """Attempt to open cameras within a range to see which indices are available."""
        available_cameras = []
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        return available_cameras

    def count_objects(self, objectType):
        path = 'llm-roboticarm/vision_data/{}.pt'.format(objectType)

        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
        cap = cv2.VideoCapture(1)
        #cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            frame = frame[0:480, 0:240]
            results = model(frame)
            coords_plus = results.pandas().xyxy[0]
            object_count = 0
            for index, row in coords_plus.iterrows():
                name = row['name']
                if name == 'housing-flat':
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    object_count += 1
                    
            if object_count > 1:
                cv2.putText(frame, f'{objectType} objects: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(f'{objectType} Objects Detection', frame)
                # Escape loop on pressing Esc key
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                cap.release()
                cv2.destroyAllWindows()
                exit()
            else:
                cap.release()
                cv2.destroyAllWindows()
                return
        
    def count_and_display_objects(self):
        path = 'llm-roboticarm/vision_data/housing.pt'

        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
        cap = cv2.VideoCapture(1)
        #cap = cv2.VideoCapture(0)
        midpoints=[]
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            frame = frame[0:480, 0:240]
            results = model(frame)
            coords_plus = results.pandas().xyxy[0]
            housing_object_count = 0
            for index, row in coords_plus.iterrows():
                name = row['name']
                if name == 'housing-flat':
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    housing_object_count += 1
                    midpoints.append([(x1+x2)/2,(y1+y2)/2])
            cv2.putText(frame, f'Housing objects: {housing_object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Housing Objects Detection", frame)
            # Escape loop on pressing Esc key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        print(midpoints)
        cap.release()
        cv2.destroyAllWindows()
        if housing_object_count>1 and len(midpoints)>1 and math.sqrt((int(midpoints[0][0]-midpoints[1][0]))**2+int((midpoints[0][1]-midpoints[1][1]))**2)<70:
                exit()

if __name__ == "__main__":
    assembly = RoboticArmAssembly()
    #assembly.cameraCheck("wedge")
    assembly.start_robotic_assembly()
    #assembly.find_available_cameras()
    #assembly.count_and_display_objects()
