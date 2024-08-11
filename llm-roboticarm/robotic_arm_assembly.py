
import os
import threading
import sys
import time
import json
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
import pathlib
from pathlib import Path
pathlib.PosixPath=pathlib.WindowsPath

import torch
import cv2
#######################################################

class RoboticArmAssembly:
    def __init__(self):
        self.arm = XArmAPI('192.168.1.240', baud_checkset=False)
        self.variables = {}
        self.a,self.base=self.arm.get_position()
        
        self.params = {
            'grip_speed':800,
            'radius':-1,
            'auto_enable':True,
            'wait':True,
            'speed': 180,
            'acc': 10000,
            'angle_speed': 20,
            'angle_acc': 500,
            'events': {},
            'variables': self.variables,
            'callback_in_thread': True,
            'quit': False,
            'default_angles':[180,0,0]
        }
        self.step_already_done = None

    def detect_qr_code_from_camera(self):
        '''
        :detect_qr_code_from_camera() is used to find the pixel coordinates of the reference qr code

        :This function does not require any arguments to be passed

        :return: pixel coordinates of the center of the qr code
        '''
        cap = cv2.VideoCapture(2)
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

    def quality_check(self, assembly_step):
        '''
        :Checks visually to determine if the previous assembly step was successful

        :assembly_step is of type str and can be 'housing','wedge','spring', or 'cap'
        '''
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
        '''
        :Determines if the specified object is in the camera's field of view

        :objectType is a string that can be 'housing', 'wedge', 'spring', or 'cap

        :return: coordinates of the midpoint of the object's bounding box or it returns that the object cannot be found
        '''
        try:
            print(self.base[0])
            print('here')
            path = 'C:/Users/adeva/Downloads/{}.pt'.format(objectType)
            # model = None
            print(objectType)
            print(path)
            
            model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local')
            print('here2')
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

    with open('C:/Users/jongh/projects/llm-roboticarm/llm-roboticarm/params.json','r') as file:
        data=json.load(file)
        data['housing_set']
        data['wedge_set']
        data['spring_set']
        data['cap_set']
        data['housing_90']
        data['wedge_90']
        data['spring_90']

    '''
    
    '''
    def movement(self,bounding_box_coords,object_set,object_90,pixel_x_coord,pixel_y_coord):
        """
        :Performs the arm's movement to detected objects, gripping operation, and movement to the assembly area

        :bounding_box_coords is an 4 element list where the fist two elements are the pixel coordinates
        of the top left corner of the bounding box and the last two are the pixel coordinates of the bottom right
        :object_set is the list containing the set of assembly instructions for each part when the object is 
        oriented vertically, located in the parameter file
        :object_90 is the list containing the set of assembly instructions for each part when the object is 
        oriented horizonatally, located in the parameter file
        :pixel_x_coord is a float/ int that is the x coordinate of the midpoint of the detected object
        :pixel_y_coord is a float/ int that is the y coordinate of the midpoint of the detected object

        """
        
        setTypes ={True: object_set, False: object_90}
        set1=setTypes[bounding_box_coords[2] - bounding_box_coords[0] <= bounding_box_coords[3] - bounding_box_coords[1]]
        if self.arm.error_code != 0 or self.params['quit']:
            
            return
        for i in self.data[set1]:
            # Determines robot function based on size and type of each element in set1
            if isinstance(i, int):
                self.arm.set_gripper_position(i, wait=self.params['wait'], speed=self.params['grip_speed'], auto_enable=self.params['auto_enable'])
            elif len(i) == 4:
                self.arm.set_position(*([pixel_x_coord, pixel_y_coord] + i), speed=self.params['speed'], mvacc=self.params['acc'], radius=self.params['radius'], wait=self.params['wait'])
            else:
                self.arm.set_position(*i, speed=self.params['speed'], mvacc=self.params['acc'], radius=self.params['radius'], wait=self.params['wait'])

            


    def vision_check(self,object_type):
        '''
        :Finds the next piece in the assembly before picking is done
        '''
        a,base=self.arm.get_position()
        self.arm.set_position(*(self.data['camera_bottom_left']+[self.params['camera_z']]+self.params['default_angles']), speed=self.params['speed'], mvacc=self.params['acc'], radius=self.params['radius'], wait=self.params['wait'])
        self.rotate(object_type)






    def rotate(self,object_type):
        a,base=self.arm.get_position()
        gripper_x,gripper_y=base[0],base[1]
        while self.objectPlace(object_type)==None:
            #If object is not found, robot moves around its base in a square path until an object is found or deemed missing
            if gripper_x < 390:
                gripper_x += 85
                code = self.arm.set_position(*[gripper_x, gripper_y, 465, 180.0, 0.0, 0.0], speed=self.params['speed'],mvacc=self.params['acc'], radius=-1.0, wait=True)
                code = self.arm.set_position(*[gripper_x, gripper_y, 465, 180.0, 0.0, 0.0], speed=self.params['speed'],mvacc=self.params['acc'], radius=-1.0, wait=True)
                self.objectPlace(object_type)
            elif gripper_y < 250:
                gripper_y += 10
                code = self.arm.set_position(*[gripper_x, gripper_y, 465, 180.0, 0.0, 0.0], speed=self.params['speed'],mvacc=self.params['acc'], radius=-1.0, wait=True)
                self.objectPlace(object_type)
            else:
                print('no object')
                exit()
    
        

    def perform_housing_step(self,coord_list):
        try: 
            self.xhouse = int(coord_list[0] + coord_list[2])
            self.xhouse = self.xhouse /2

            self.yhouse = int(coord_list[1] + coord_list[3])
            self.yhouse = self.yhouse /2

            self.xH = self.base[0] + (self.yhouse)*-0.6+235
            self.yH = self.base[1] + (self.xhouse)*-0.6+215            
        except:
            message = f"Error: the housing object was not detected"
            return message

        try:
            
            self.movement(coord_list,"housing_set","housing_90",self.xH,self.yH)
        except Exception as e:
            message = f"Error {e}: there was an error during the housing movement"
            return message
                    
        self.step_already_done = "housing"
        message = "Housing step completed successfully."
        return message          

        
    def perform_wedge_step(self,coord_list):
        try:
            self.xwedge = int(coord_list[0] + coord_list[2])
            self.xwedge = self.xwedge /2

            self.ywedge = int(coord_list[1] + coord_list[3])
            self.ywedge = self.ywedge /2

            self.xW = self.base[0] + (self.ywedge)*-0.6+245
            self.yW = self.base[1] + (self.xwedge)*-0.6+230
        except:
            return f"Error: during wedge object placement detection"

        try:
            self.movement(coord_list,"wedge_set","wedge_90",self.xW,self.yW)
        except Exception as e:
            return f"Error {e} during wedge movement"
                        
        #try:
            #self.quality_check("wedge")
        #except:
            #### temporary for test ####
            #self.step_already_done = "wedge"
            ############################
            #return f"Error: wedge object placement is not done correctly."
    
        self.step_already_done = "wedge"
        
        return "Wedging step completed successfully."
                
    def perform_spring_step(self,coord_list):
        try:
            self.xspring = int(coord_list[0] + coord_list[2])
            self.xspring = self.xspring / 2
            # # #
            self.yspring = int(coord_list[1] + coord_list[3])
            self.yspring = self.yspring / 2
            #
            self.xS = self.base[0] + (self.yspring)*-0.6+235
            self.yS = self.base[1] + (self.xspring)*-0.6+230
        except:
            message = f"Error: the spring object was not detected"
            return message

        try:
            self.movement(coord_list,"spring_set","spring_90",self.xS,self.yS)
        except Exception as e:
            return f"Error {e} during spring movement"
                        
        self.step_already_done = "spring"
        return "Spring step completed successfully."

    def perform_cap_step(self,coord_list):
        try:
            self.xcap = int(coord_list[0] + coord_list[2])
            self.xcap = self.xcap / 2

            self.ycap = int(coord_list[1] + coord_list[3])
            self.ycap = self.ycap / 2

            self.xC = self.base[0] + (self.ycap)*-0.6+247
            self.yC = self.base[1] + (self.xcap)*-0.6+230
        except:
            return f"Error: {e} during cap object placement detection"

        try:
            self.movement(coord_list,"cap_set","cap_90",self.xC,self.yC)
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
        '''
        :Counts total objects of type objectType

        :objectType is of type str where it can be 'housing','wedge','spring', or 'cap'
        '''
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
        


    def count_and_display_objects(self):
        '''
        Displays a frame with bounding boxes around all objects and a count for how many there are.
        Additionally, provides coordinates of each type of object
        '''
        path = 'llm-roboticarm/vision_data/combined.pt'

        # Loads the model
        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local', force_reload=True)
        
        # Opens the camera
        cap = cv2.VideoCapture(2)  # Change to 0 if you want to use the default camera
        
        object_counts = {}
        object_coords = {}
        temp=0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 480))
        
            results = model(frame)
            coords_plus = results.pandas().xyxy[0]
            
            # Reset the counts and coordinates for each frame
            object_counts.clear()
            object_coords.clear()
            
            for index, row in coords_plus.iterrows():
                name = row['name']
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                
                #Makes the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # counts the objects and stores the coordinates
                object_counts[name] = object_counts.get(name, 0) + 1
                object_coords[name] = [x1, y1, x2, y2]
            
            # Displays the number of each object
            y_offset = 30
            for obj_name, count in object_counts.items():
                cv2.putText(frame, f'{obj_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 20
            
      
            cv2.imshow("Object Detection", frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == 27:  #Can end loop by pressing the Esc key
                break
            
            '''
            for obj_name, coords in object_coords.items():
                if obj_name=='housing' and temp==0:
                    self.perform_housing_step(coords)
                elif obj_name=='wedge' and temp==1:
                    self.perform_wedge_step(coords)
                elif obj_name=='spring' and temp==2:
                    self.perform_spring_step(coords)
                elif obj_name=='cap' and temp==3:
                    self.perform_cap_step(coords)
            temp+=1 
            '''
            
            if temp < len(self.assembly_steps):
                obj_name = self.assembly_steps[temp]
                if obj_name in object_coords:
                    coords = object_coords[obj_name]
                    getattr(self, f"perform_{obj_name}_step")(coords)
                    temp += 1
            

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    
    assembly = RoboticArmAssembly()
    #assembly.quality_check("wedge")
    #assembly.start_robotic_assembly()
    #assembly.find_available_cameras()
    assembly.count_and_display_objects()
    