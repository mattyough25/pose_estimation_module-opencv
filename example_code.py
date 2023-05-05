import cv2
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture('videos/Dancing_Man.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()  
    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,
            (255,0,0),3)
    pTime = cTime
    
    scale_percent = 25  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    imgToShow = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", imgToShow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break