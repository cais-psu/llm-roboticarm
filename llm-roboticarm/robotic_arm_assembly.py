
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
import llm_agent
from pyzbar.pyzbar import decode
import json
from voice_control import VoiceControl
import logging
import openai
from rag_handler import RAGHandler
from prompts import VERBAL_UPDATES_INSTRUCTIONS
import asyncio

#######################################################

class RoboticArmAssembly:
    """
    A class to represent the robotic arm assembly process.
    """    
    def __init__(self, params_json):
        ############################ Set up agent-specific logger ############################
        self.logger = logging.getLogger(f'agent_xArm')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        file_handler = logging.FileHandler(f'llm-roboticarm/log/xArm_actions.log', mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        ######################################################################################

        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.sop_handler = RAGHandler('llm-roboticarm/initialization/robots/specification/xArm_SOP.pdf', 'pdf', self.openai_api_key)

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
        self.step_working_on = None
        self.voice_control = VoiceControl()

        # Load configuration from the JSON file
        if isinstance(params_json, str):
            self.params_json = json.loads(params_json)
        else:
            self.params_json = params_json
        

    def detect_qr_code_from_camera(self):
        """
        Detects a QR code using the camera and returns the center coordinates of the QR code.

        Returns
        -------
        tuple
            A tuple containing the x and y coordinates of the center of the detected QR code.
        """        
        cap = cv2.VideoCapture(2)
        while True:
            ret, frame = cap.read()
            decoded_objects = decode(frame)
            for obj in decoded_objects:
                (self.x, self.y, self.w, self.h) = obj.rect
                cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
            cv2.imshow("QR Code Detection", frame)
            #time.sleep(1)
            #break
            if cv2.waitKey(1) & 0xFF == 27:
                break

        
        return (self.x+self.w//2,self.y+self.w//2)

    def cameraCheck(self, assembly_step):
        """
        Checks the camera for the specified assembly step using a trained model.

        Parameters
        ----------
        assembly_step : str
            The assembly step to be checked.
        """        
        path = 'llm-roboticarm/vision_data/check.pt'
        
        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
        cap = cv2.VideoCapture(2)
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
        """
        Detects and places an object using the camera and a trained model.

        Parameters
        ----------
        objectType : str
            The type of object to be placed (e.g., 'housing', 'wedge', 'spring').

        Returns
        -------
        tuple
            Coordinates and mask information of the detected object.
        """        
        try:
            path = 'llm-roboticarm/vision_data/{}.pt'.format(objectType)
            # model = None
            #print(objectType)
            #print(path)

            model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
            
            cap = cv2.VideoCapture(2)
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
        """
        Distinguishes the orientation of an object in the image using edge detection.

        Parameters
        ----------
        image : ndarray
            The image in which the orientation needs to be distinguished.

        Returns
        -------
        float
            The mean orientation of the edges in the image.
        """        
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
        """
        Pretty print for logging with timestamp and line number.

        Parameters
        ----------
        *args : list
            Arguments to print.
        **kwargs : dict
            Keyword arguments to print.
        """        
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                    ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    def display_video(self, index, name, x, y):
        """
        Displays video from the specified camera index.

        Parameters
        ----------
        index : int
            Camera index to capture video from.
        name : str
            Window name for the video display.
        x : int
            X-coordinate of the window position.
        y : int
            Y-coordinate of the window position.
        """        
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
        """
        Executes the movement for placing the housing part using the robotic arm.
        """        
        if self.x2h - self.x1h <= self.y2h - self.y1h:  # THIS IS FOR HOUSING BEFORE ELSE
            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(600, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xH, self.yH, 190.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xH, self.yH, 185.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
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

    def wedge_movement(self):
        """
        Executes the movement for placing the wedge part using the robotic arm.
        """        
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
                code = self.arm.set_position(*[self.xW,self.yW , 183, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                code = self.arm.set_position(*[self.xW, -291, 225, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[94, -280, 270, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-251.1, -334.9, 100, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-249.5, -334.9, 58, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[-81.8, -330.8, 250, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_position(*[240, -340, 240, 180.0, 0.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

    def spring_movement(self):
        """
        Executes the movement for placing the spring part using the robotic arm.
        """        
        if self.x2s - self.x1s <= self.y2s - self.y1s:
            if self.arm.error_code == 0 and not self.params['quit']:
                code = self.arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[self.xS, self.yS, 195.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[self.xS, self.yS, 181.2, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
                code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                                auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
                code = self.arm.set_position(*[self.xS, -291, 225, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
                code = self.arm.set_position(*[94, -280, 270, 180.0, -90.0, 0.0], speed=self.params['speed'], mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-249.6, -334.2, 100, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)
                code = self.arm.set_position(*[-249.6, -334.2, 58.8, 180.0, -90.0, 0.0], speed=50,
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

                code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

                code = self.arm.set_position(*[-81.8, -330.8, 250, 180.0, -90.0, 0.0], speed=self.params['speed'],
                                        mvacc=self.params['acc'],
                                        radius=-1.0, wait=True)

    def cap_movement(self):
        """
        Executes the movement for placing the cap part using the robotic arm.
        """        
        if self.arm.error_code == 0 and not self.params['quit']:
            code = self.arm.set_gripper_position(400, wait=True, speed=800, auto_enable=True)

            code = self.arm.set_position(*[self.xC, self.yC, 195.5, 180.0, 0.0, 0.0], speed=self.params['speed'],
                                    mvacc=self.params['acc'],  # For wedge use y-10 instead of y
                                    radius=-1.0, wait=True)
            code = self.arm.set_position(*[self.xC, self.yC, 183, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)

            # code = self.arm.set_gripper_position(10, wait=True, speed=200, auto_enable=True)  # This is for the spring
            code = self.arm.set_gripper_position(1, wait=True, speed=800,
                                            auto_enable=True)  # 300 for housing, 15 for wedge, 1 for spring
            code = self.arm.set_position(*[289.4, -27.6, 237, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            # code = self.arm.set_gripper_position(15, wait=True, speed=800, auto_enable=True)         #This is for the wedge
            code = self.arm.set_position(*[289.4, -322, 237, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_position(*[-64.5, -335.8, 237, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_position(*[-64.5, -335.8, 222, 180.0, 0.0, 0.0], speed=75, mvacc=self.params['acc'],
                                    radius=-1.0, wait=True)
            code = self.arm.set_gripper_position(300, wait=True, speed=800, auto_enable=True)

            if code != 0:
                self.params['quit'] = True
                self.pprint('set_position, code={}'.format(code))

        self.params['quit'] = True
        self.pprint('set_position, code={}'.format(code))

        if code != 0:
            self.params['quit'] = True
            self.pprint('set_position, code={}'.format(code))

    def perform_housing_step(self):
        """
        Performs the housing step of the assembly process.

        Returns
        -------
        str
            A message indicating the status of the housing step.
        """        
        self.step_working_on = "housing"

        try:
            self.count_and_display_housing()
        except:
            message = f"Error: the housing object is overlapping"
            self.logger.error(message)
            return message            
        
        try:
            self.x1h, self.y1h, self.x2h, self.y2h, a, b, c = self.objectPlace('housing')
            self.logger.info("housing object identified")
            self.xhouse = int(self.x1h + self.x2h)
            self.xhouse = self.xhouse /2

            self.yhouse = int(self.y1h + self.y2h)
            self.yhouse = self.yhouse /2

            self.xH = self.xQRarm + (self.yhouse - self.yQRPix)*0.65-15
            self.yH = self.yQRarm + (self.xhouse - self.xQRPix) * 0.64            
        except:
            message = f"Error: the housing object was not detected"
            self.logger.error(message)
            return message

        try:
            self.housing_movement()
        except Exception as e:
            message = f"Error {e}: there was an error during the housing movement"
            self.logger.error(message)
            return message
                    
        message = "Housing step completed successfully."
        return message          

    def perform_wedge_step(self):
        """
        Performs the wedge step of the assembly process.

        Returns
        -------
        str
            A message indicating the status of the wedge step.
        """        
        self.step_working_on = "wedge"

        try:
            self.x1w, self.y1w, self.x2w, self.y2w, mask, mask1, mask2 = self.objectPlace('wedge')

            self.xwedge = int(self.x1w + self.x2w+480)
            self.xwedge = self.xwedge /2

            self.ywedge = int(self.y1w + self.y2w)
            self.ywedge = self.ywedge /2

            self.xW = self.xQRarm + (self.ywedge - self.yQRPix) * 0.65
            self.yW = self.yQRarm + (self.xwedge - self.xQRPix) * 0.64
        except:
            error_message = f"Error: during wedge object placement detection"
            self.logger.error(error_message)
            return error_message

        try:
            self.wedge_movement()
        except Exception as e:
            error_message = f"Error {e} during wedge movement"
            self.logger.error(error_message)            
            return error_message
                        
        #try:
            #self.cameraCheck("wedge")
        #except:
            #### temporary for test ####
            #self.step_working_on = "wedge"
            ############################
            #return f"Error: wedge object placement is not done correctly."
        message = "Wedging step completed successfully."
        self.logger.info(message)
        return message
                
    def perform_spring_step(self):
        """
        Performs the spring step of the assembly process.

        Returns
        -------
        str
            A message indicating the status of the spring step.
        """
        self.step_working_on = "spring"        
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
            self.logger.error(message)
            return message

        try:
            self.spring_movement()
        except Exception as e:
            error_message = f"Error {e} during spring movement"
            self.logger.error(error_message)
            return error_message

        message = "Spring step completed successfully."
        self.logger.info(message)
        return message

    def perform_cap_step(self):
        """
        Performs the cap step of the assembly process.

        Returns
        -------
        str
            A message indicating the status of the cap step.
        """        
        self.step_working_on = "cap"

        try:
            self.x1c, self.y1c, self.x2c, self.y2c, a, b, c = self.objectPlace('cap')

            self.xcap = int(self.x1c + self.x2c+880)
            self.xcap = self.xcap / 2

            self.ycap = int(self.y1c + self.y2c)
            self.ycap = self.ycap / 2

            self.xC = self.xQRarm + (self.ycap - self.yQRPix) * 0.65
            self.yC = self.yQRarm + (self.xcap - self.xQRPix) * 0.64
        except:
            error_message = f"Error: {e} during cap object placement detection"
            self.logger.error(error_message)
            return error_message

        try:
            self.cap_movement()
        except Exception as e:
            error_message = f"Error {e} during cap movement"
            self.logger.error(error_message)
            return error_message
                        
        self.step_working_on = "completed"
        message = "Cap step completed successfully."
        self.logger.info(message)
        return message

    def _verbal_updates(self, step_working_on: str):
        """
        Retrieve safety information for starting the process and provide verbal updates.
        """
        message = self.sop_handler.retrieve(f"Assembly step working on: {step_working_on}." + VERBAL_UPDATES_INSTRUCTIONS)
        threading.Thread(target=self.voice_control.text_to_speech, args=(message, 0)).start()

    def robotic_assembly(self, step_working_on: str):
        """
        Starts the robotic assembly process by performing each step sequentially.
        If the step is not part of the assembly steps, it starts from the beginning.

        :param step_working_on: name of the assembly step that is being worked on
        """

        # Run verbal updates asynchronously
        threading.Thread(target=self._verbal_updates, args=(step_working_on,)).start()

        # Define the order of assembly steps
        # assembly_steps = ["housing", "wedge", "spring", "cap"]
        assembly_steps = self.params_json.get("assembly_steps", [])

        # Determine if the process should start from the beginning
        start_from_beginning = step_working_on not in assembly_steps

        # Voice feedback for starting/resuming the process
        if start_from_beginning:
            self.logger.info("Starting the robotic assembly process from the beginning")
            step_working_on = assembly_steps[0]  # Start from the first step

            #threading.Thread(self.voice_control.text_to_speech("Sure! First, let me go through calibration process. Press ESC when calibration is completed.")).start()
            self.xQRPix, self.yQRPix = self.detect_qr_code_from_camera()
            self.xQRarm = 293
            self.yQRarm = -130

            #threading.Thread(self.voice_control.text_to_speech("The robot is now preparing to execute the assembly process. For your safety, please keep a safe distance from the robot.")).start()
        else:
            self.logger.info(f"Starting the assembly operation from the {assembly_steps} step")
            #threading.Thread(self.voice_control.text_to_speech("Certainly! Resuming the assembly process from where I left off!")).start()

        # Get the current index of the step
        current_index = assembly_steps.index(step_working_on)

        # Perform the assembly steps
        for step in assembly_steps[current_index:]:
            #threading.Thread(self.voice_control.text_to_speech("Performing " + step + " assembly process.")).start()
            self.logger.info(f"Performing the {step} step")
            message = getattr(self, f"perform_{step}_step")()

            if 'error' in message.lower():
                return self.step_working_on, message

        self.step_working_on = "completed"
        message = "All steps for the assembly are successfully completed!"
        self.logger.info(message)
        return self.step_working_on, message

    '''
    def resume_operation_from_last_step(self, step_working_on: str):
        """
        Resumes the assembly process from the last completed step.

        :param step_working_on: name of the assembly step that is working on
        :param sop_handler: parses 'self.sop_handler' class that contains information on SOP
        """
        self.logger.info("Resuming assembly operation from the last step")
        threading.Thread(self.voice_control.text_to_speech("Certainly! Resuming the assembly process from where I left off!")).start()

        # Define the order of assembly steps
        #assembly_steps = self.params_json.get("assembly_steps", [])
        assembly_steps = ["housing", "wedge", "spring", "cap"]
        current_index = assembly_steps.index(step_working_on) if step_working_on in assembly_steps else -1
        
        for step in assembly_steps[current_index:]:
            threading.Thread(self.voice_control.text_to_speech("Performing " + step + " assembly process.")).start()
            message = getattr(self, f"perform_{step}_step")()
            if 'error' in message.lower():
                return self.step_working_on, message

        self.step_working_on = "completed"
        return self.step_working_on, "All steps for the assembly are successfully completed."
    
    def start_robotic_assembly(self, step_working_on: str):
        """
        Starts the robotic assembly process by performing each step sequentially.

        :param step_working_on: name of the assembly step that is working on
        """      
        message = self.sop_handler.retrieve("retrieve safety information for starting robotic assembly")
        print(message)

        threading.Thread(self.voice_control.text_to_speech("Sure! First, let me go through calibration process. Press ESC wehn calibration is completed.")).start()

        self.xQRPix,self.yQRPix = self.detect_qr_code_from_camera()
        self.xQRarm=293
        self.yQRarm=-130

        threading.Thread(self.voice_control.text_to_speech("The robot is now preparing to execute cable shark assembly process. For your safety, please keep a safe distance from the robot.")).start()
        
        for step in self.params_json.get("assembly_steps", []):
            threading.Thread(self.voice_control.text_to_speech("Performing " + step + " assembly process.")).start()
            message = getattr(self, f"perform_{step}_step")()
            print(message)
            if 'error' in message.lower():
                return self.step_working_on, message

        self.step_working_on = "completed"
        message = "Cable shark assembly process is completed successfully!"

        return self.step_working_on, message
    '''
    def find_available_cameras(self):
        """
        Attempts to open cameras within a range to see which indices are available.

        Returns
        -------
        list
            A list of available camera indices.
        """        
        """Attempt to open cameras within a range to see which indices are available."""
        available_cameras = []
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        return available_cameras

    def count_objects(self, objectType):
        """
        Counts the number of objects of a specified type using the camera and a trained model.

        Parameters
        ----------
        objectType : str
            The type of object to count.

        Returns
        -------
        None
        """        
        path = 'llm-roboticarm/vision_data/{}.pt'.format(objectType)

        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
        cap = cv2.VideoCapture(2)
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
        
    def count_and_display_housing(self):
        """
        Counts and displays the housing objects using the camera and a trained model.

        Returns
        -------
        None
        """        
        path = 'llm-roboticarm/vision_data/housing.pt'

        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
        #cap = cv2.VideoCapture(1)
        cap = cv2.VideoCapture(2)
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
        #print(midpoints)
        cap.release()
        cv2.destroyAllWindows()
        if housing_object_count>1 and len(midpoints)>1 and math.sqrt((int(midpoints[0][0]-midpoints[1][0]))**2+int((midpoints[0][1]-midpoints[1][1]))**2)<70:
                exit()

if __name__ == "__main__":
    params_file = 'llm-roboticarm/initialization/robots/specification/params.json'
    with open(params_file, 'r') as file:
        params_information = json.load(file)

    assembly = RoboticArmAssembly(params_information)
    #assembly.cameraCheck("wedge")
    #assembly.robotic_assembly("none")
    assembly.robotic_assembly(step_working_on="None")

    #assembly.count_and_display_housing()
