import cv2
import numpy as np
import math
from eye_tracker import contouring, eye_on_mask, print_eye_pos, process_thresh
from face_detector import get_face_detector, find_faces
from face_landmarks import draw_marks, get_landmark_model, detect_marks
from head_pose_estimation import head_pose_points

EyetrackerList=[]
MouthtrackerList=[]
HeadtrackerList=[]

face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

fps = int(cap.get(cv2.CAP_PROP_FPS))
while True:

    ret, img = cap.read()
    rects = find_faces(img, face_model)
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        draw_marks(img, shape)
        cv2.putText(img, 'Press r to start processing', (30, 30), font,
                    1, (0, 255, 255), 2)
        cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        for i in range(100):
            for i, (p1, p2) in enumerate(outer_points):
                d_outer[i] += shape[p2][1] - shape[p1][1]
            for i, (p1, p2) in enumerate(inner_points):
                d_inner[i] += shape[p2][1] - shape[p1][1]
        break
cv2.destroyAllWindows()
d_outer[:] = [x / 100 for x in d_outer]
d_inner[:] = [x / 100 for x in d_inner]



cv2.namedWindow('image')
thresh = img.copy()
kernel = np.ones((9, 9), np.uint8)
def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)


while True:
    ret, img = cap.read()
    if ret == True:
        
        #face_detector
        faces = find_faces(img, face_model)
        if(faces.__len__()==1):
            head_pose='Legal'
            for face in faces:
                marks = detect_marks(img, landmark_model, face)
                image_points = np.array([
                                        marks[30],     # Nose tip
                                        marks[8],     # Chin
                                        marks[36],     # Left eye left corner
                                        marks[45],     # Right eye right corne
                                        marks[48],     # Left Mouth corner
                                        marks[54]      # Right mouth corner
                                    ], dtype="double")
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90
                     
                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90
                    # print('div by zero error')
                    
                if ang1 >= 45:
                    HeadtrackerList.append('Head down')
                    head_pose='Not Legal'
                    cv2.putText(img, 'Head down', (30, 30), font, 1, (0, 255, 255), 2)
                elif ang1 <= -48:
                    HeadtrackerList.append('Head up')
                    cv2.putText(img, 'Head up', (30, 30), font, 1, (0, 255, 255), 2)
                    head_pose='Not Legal'
                elif ang2 >= 40:
                    HeadtrackerList.append('Head right')
                    cv2.putText(img, 'Head right', (30, 30), font, 1, (0, 255, 255), 2)
                    head_pose='Not Legal'
                elif ang2 <= -40:
                    HeadtrackerList.append('Head left')
                    cv2.putText(img, 'Head left', (30, 30), font, 1, (0, 255, 255), 2)
                    head_pose='Not Legal'
                else:
                    HeadtrackerList.append('Legal')
                    
                    
                cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 2)
                cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 2)
            if head_pose=='Legal':   
                
                #mouth_opening_detector
                
                rects = find_faces(img, face_model)
                for rect in rects:
                    shape = detect_marks(img, landmark_model, rect)
                    cnt_outer = 0
                    cnt_inner = 0
                    draw_marks(img, shape[48:])
                    for i, (p1, p2) in enumerate(outer_points):
                        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                            cnt_outer += 1 
                    for i, (p1, p2) in enumerate(inner_points):
                        if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                            cnt_inner += 1
                    if cnt_outer > 3 and cnt_inner > 2:
                        MouthtrackerList.append('Mouth open')
                        cv2.putText(img, 'Mouth open', (30, 70), font,1, (0, 255, 255), 2)
                    else:
                        MouthtrackerList.append('Legal')
                    
                    
            if head_pose=='Legal':  
                  
            #eye_tracker
                rects = find_faces(img, face_model)
                for rect in rects:
                    shape = detect_marks(img, landmark_model, rect)
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    mask, end_points_left = eye_on_mask(mask, left, shape)
                    mask, end_points_right = eye_on_mask(mask, right, shape)
                    mask = cv2.dilate(mask, kernel, 5)
                    eyes = cv2.bitwise_and(img, img, mask=mask)
                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]
                    mid = int((shape[42][0] + shape[39][0]) // 2)
                    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                    thresh = process_thresh(thresh)
                    eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                    eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                    eyetext=print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
                    EyetrackerList.append(eyetext)
                
                
            

            
                    
                    
            cv2.imshow("image", thresh)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
        
                break
        elif (faces.__len__()==0):
            cv2.putText(img, 'No face Detected', (30, 150), font,1, (0, 255, 255), 2)
            MouthtrackerList.append('No face')
            EyetrackerList.append('No face')
            HeadtrackerList.append('No face')
            print('No Face')
        else:
            cv2.putText(img, 'Multiple faces Detected', (30, 150), font,1, (0, 255, 255), 2)
            MouthtrackerList.append('Multiple faces detetced')
            EyetrackerList.append('Multiple faces detetced')
            HeadtrackerList.append('Multiple faces detetced')
            print('Multiple faces detected')
    else:
        break



NoOfFramesRecorded=min(len(EyetrackerList),len(HeadtrackerList),len(MouthtrackerList))
SecondWIseValidationList=[]
NoOfSeconds=0
for i in range(0,NoOfFramesRecorded+1,fps):
    FrameValidationList=[]
    maxlim=min(i+fps,NoOfFramesRecorded)
    NoOfSeconds=NoOfSeconds+1
    for j in range(i,maxlim):
        if(EyetrackerList[j]!='Legal'or HeadtrackerList[j]!='Legal' or MouthtrackerList[j]!='Legal'):
            FrameValidationList.append('Illegal')
        else:
            FrameValidationList.append('Legal')
    if(FrameValidationList.count('Illegal')>fps//3):
        SecondWIseValidationList.append('Illegal')
    else:
        SecondWIseValidationList.append('Legal')
        
        
           

print(str(SecondWIseValidationList.count('Illegal')) +' seconds of Suspicious activity found in total duration of '+  str(NoOfSeconds)+' seconds' )           
cv2.destroyAllWindows()
cap.release()