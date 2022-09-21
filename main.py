from asyncio.windows_events import NULL
import enum
from turtle import color
from unittest import result
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

######################################
#end importation 
######################################


#### functions ######
def calc_angle(a,b,c):
    a=np.array(a) #first
    b=np.array(b) #middle
    c=np.array(c) #end

    radiants=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angel=np.abs(radiants*180/np.pi)

    if angel> 180:
        angel = 360 - angel
    
    return angel
##### set ups #######

mp_drawing = mp.solutions.drawing_utils
mp_hands= mp.solutions.pose
stage = None
counter = 0

########starting cam

cap = cv2.VideoCapture(0)

with mp_hands.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        
    while cap.isOpened():
        ret,frame=cap.read()
        image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results=hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # if results.multi_pose_landmarks:
        #     for num , hand in enumerate(results.multi_pose_landmarks):
        #         mp_drawing.draw_landmarks(image,hand,mp_hands.POSE_CONNECTIONS)

        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_hands.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(153,0,0),thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(102,0,0),thickness=2, circle_radius=2),
                                    )
         
        try:
            landmarksres= results.pose_landmarks.landmark
            # print( len(landmarksres) )
            # print(landmarksres[mp_hands.PoseLandmark.LEFT_SHOULDER.value].x)
            shoulder = [landmarksres[mp_hands.PoseLandmark.LEFT_SHOULDER.value].x,landmarksres[mp_hands.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarksres[mp_hands.PoseLandmark.LEFT_ELBOW.value].x,landmarksres[mp_hands.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarksres[mp_hands.PoseLandmark.LEFT_WRIST.value].x,landmarksres[mp_hands.PoseLandmark.LEFT_WRIST.value].y]
            angel = calc_angle(shoulder,elbow,wrist)
            
            
            if angel > 150 :
                stage = "down"
             

            if angel < 40 and stage == "down": 

                cv2.putText(image, str(angel),tuple(np.multiply(elbow,[640,480]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2, cv2.LINE_AA)
                stage = "up"
                counter +=1
            else:
                
                    
                cv2.putText(image, str(angel),tuple(np.multiply(elbow,[640,480]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2, cv2.LINE_AA)
                
            
            #render curl counter
            cv2.rectangle(image, (0,0),(255,73),(254,117,16),-1)
            #reps 
            cv2.putText(image, "REPS",(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),1,cv2.LINE_AA)

            cv2.putText(image, str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255),2,cv2.LINE_AA)
            
        except:
            
            pass

        ### organs that we need to calc angel for biceps curl ###

        
   

        # print(landmarksres[mp_hands.PoseLandmark.NOSE])

        cv2.imshow('pose tracking',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


 
        
cap.release()
cv2.destroyAllWindows()