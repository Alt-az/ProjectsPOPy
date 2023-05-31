import cv2
import numpy as np
import time
import math
def getAngle(pointsList,frame):
    b,a,c = pointsList
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    print(a,b,c,ang)
    if ang < 0 :
        ang=ang*-1
    cv2.putText(frame,str(round(ang)),(b[0]-40,b[1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)
def findClosest(array,point):
    result = (0,0)
    mindist = 1000000
    for b in array:
        dist = ((b[0] - point[0])**2 + (b[1] - point[1])**2)**0.5
        if mindist>dist:
            mindist = dist
            result = b
    print(mindist)
    return result,mindist

cap = cv2.VideoCapture('short.mp4')
pos_frame = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)

break_flag=False
start_flag=True
previous_frame = None
maks = 0
x1 = 0
y1 = 0
r1 = 0

while True:
    t = time.time()
    for i in range(25):
        flag, frame = cap.read()
        if flag:
            todisplay = frame
            obraz_sz = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obraz_fil = cv2.GaussianBlur(obraz_sz, (9, 9), 0)
            black= np.zeros(obraz_fil.shape)
            if(start_flag):
                bbox = cv2.selectROI("Obraz", frame, False)
                cv2.setWindowProperty('Obraz', cv2.WND_PROP_TOPMOST, 1)
                cv2.moveWindow('Obraz', 20, 20)
                start_flag = False
                tracker = cv2.legacy.TrackerCSRT_create()
                pt2 = [0, 0]
                pt2[0], pt2[1], stw, sth = bbox
                tracker.init(frame, bbox)

            circles=cv2.HoughCircles(obraz_fil, cv2.HOUGH_GRADIENT,
              dp=1,
              minDist=5,
              param1=150,
              minRadius=0,
              maxRadius=0)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    if(r>maks):
                        maks=r
                        x1=x
                        y1=y
                        r1=r
                cv2.circle(todisplay,(x1,y1),r1,(255, 0, 0),2)

                if (previous_frame is None):
                    previous_frame = obraz_fil
                    continue
                diff_frame = cv2.absdiff(src1=previous_frame, src2=obraz_fil)
                previous_frame = obraz_fil
                kernel = np.ones((5, 5))

                diff_frame = cv2.dilate(diff_frame, kernel, 1)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

                kierownica = black.astype('uint8')
                kierownica[y1 - r1:(y1 + 2 * r1), x1 - r1:(x1 + 2 * r1)] = thresh_frame[y1 - r1:(y1 + 2 * r1),
                                                                           x1 - r1:(x1 + 2 * r1)]

                contours, _ = cv2.findContours(image=kierownica, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # Calculate area and remove small elements
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        cv2.drawContours(todisplay, [cnt], -1, (0, 255, 0), 2)
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    lastbbox= bbox
                else:
                    array = []
                    for cnt in contours:
                        array.append(cv2.boundingRect(cnt))
                    if array != []:
                        mdist = findClosest(array, bbox)
                        bbox = mdist[0]
                    else:
                        bbox=lastbbox
                    if len(bbox) == 2:
                        bbox = (bbox[0], bbox[1], stw, sth)
                    tracker.init(obraz_fil, bbox)
                # getAngle(((x1, y1), (pt2[0],pt2[1]),(bbox[0],bbox[1])), todisplay)
                x,y,_,_=bbox
                # cv2.line(todisplay,  (int(x1),int(y1)),(int(x),int(y)), (255, 255, 255), 5)

            cv2.imshow("Obraz", todisplay)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print ("ramka nie jest gotowa")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break_flag = True
            break
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break_flag=True
            break
    t = time.time()-t
    if break_flag:
        cap.release()
        break
    print(int(25/t))