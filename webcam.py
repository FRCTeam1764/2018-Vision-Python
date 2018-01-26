import numpy as np
import cv2
import math
windowName = 'settings'

def nothing(x):
    pass

def calcDistance(w):
    if w == 0:
        return 1
    else:
        return (1.083 * 554.017)/(w)

def calcWidth(p, dist):
    return (dist * p) / 554.017

def calcAngle(x, dist, camWidth):
    d = x - (camWidth/2)
    aw = calcWidth(d, dist)

    
    
    return math.degrees(math.atan(aw/dist))
    

class CameraTracker:
    def __init__(self, name=None):
        self.headless = name is None
        self.cap = cv2.VideoCapture(0)
        self.name = name
        if not self.headless:
            cv2.namedWindow(name)
			
			## TODO: Make config for setting trackbar position or maybe a thing for creating paramaters
            cv2.createTrackbar('LH', name,0,255,nothing)
            cv2.createTrackbar('LS', name,0,255,nothing)
            cv2.createTrackbar('LV', name,0,255,nothing)

            cv2.createTrackbar('HH', name,0,255,nothing)
            cv2.createTrackbar('HS', name,0,255,nothing)
            cv2.createTrackbar('HV', name,0,255,nothing)
            cv2.createTrackbar('epsilon', name,0,1000,nothing)
            cv2.setTrackbarPos('LH', name, 20)
            cv2.setTrackbarPos('LS', name, 81)
            cv2.setTrackbarPos('LV', name, 230)
            cv2.setTrackbarPos('HH', name, 39)
            cv2.setTrackbarPos('HS', name, 238)
            cv2.setTrackbarPos('HV', name, 255)
        self.kernel = np.ones((10,10), np.uint8)

    def getTrackVal(self, name):
        return cv2.getTrackbarPos(name, self.name)
	
    def processingPipeline(self, frame, lower_limit, upper_limit):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
        res = cv2.bitwise_and(frame,frame, mask=mask)
        erosion = cv2.erode(mask, self.kernel, iterations = 2)
        dilation = cv2.dilate(erosion, self.kernel, iterations = 1)
        im2, contours, heirarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        finalRes = cv2.drawContours(frame, contours, -1, (0,255,0), 10)
        #finalRes = frame
        largestRect = (0,0,0,0)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            epsilon = (self.getTrackVal('epsilon')/1000.0)*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            print(cnt)
            print("DDD")
            cv2.polylines(finalRes, [approx], True, (0,0,255), 3)
            if (w*h) >= largestRect[2]*largestRect[3]:
                largestRect = (x,y,w,h)

        x,y,w,h = largestRect
        cv2.rectangle(finalRes, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(finalRes, '{0}, {1} ::: {2} :::'.format(w,h, calcAngle(x+(w/2),calcDistance(w),frame.shape[1])), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
            #print("{0}, {1}".format(w,h))
        return finalRes
	
    def run(self):
        lh, ls, lv, hh, hs, hv = (0,0,0,0,0,0)

        while True:
            if not self.headless:
                lh = self.getTrackVal('LH')
                ls = self.getTrackVal('LS')
                lv = self.getTrackVal('LV')
                hh = self.getTrackVal('HH')
                hs = self.getTrackVal('HS')
                hv = self.getTrackVal('HV')
                ret, frame = self.cap.read()
                lower_limit = np.array([lh,ls,lv])
                upper_limit = np.array([hh,hs,hv])
                f = self.processingPipeline(frame,lower_limit,upper_limit) 
                cv2.imshow(self.name, f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows()

ct = CameraTracker(name='test')
ct.run()

