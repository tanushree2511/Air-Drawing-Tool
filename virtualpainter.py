import cv2
import os
import handtrackingmodule as htm
import numpy as np

brushThickness = 10
eraserThickness= 100

# Folder path for header images
folderPath= "header"
myList= os.listdir(folderPath)
print(myList)

# Overlay images list
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

# Set header image
header = overlayList[0]
drawColor=(0,0,255)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width of webcam frame (original)
cap.set(4, 480)  # Set height of webcam frame (original)

# Resize header to 1280x720
header = cv2.resize(header, (1280, 720))

detector= htm.handDetector(detectionCon=0.85)
xp,yp = 0,0

imgCanvas = np.zeros((720,1280,3),np.uint8)
while True:
    #1.import image
    success, img = cap.read()
    img=cv2.flip(img,1)#flipping image
    if not success:
        break
    #2. find hand landmarks
    img=detector.findHands(img)
    lmList= detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        print(lmList)
        #tip of index and middle fingers
        x1,y1= lmList[8][1:]
        x2,y2= lmList[12][1:]


        

        #3.check which fingers are up

        fingers = detector.fingersUp()
        print(fingers)
    #4.if selection mode - two fingers up 
        if fingers[1] and fingers[2]:
            xp ,yp =0,0
            print("selection mode")
            #checking for the click
            if y1 < 125:
                if 50<x1<250:
                    header = overlayList[0]
                    drawColor=(0,0,255)
                elif 350<x1<550:
                    header = overlayList[1]
                    drawColor=(255,0,0)
                elif 600<x1<750:
                    header = overlayList[2]
                    drawColor=(0,255,0)
                elif 750<x1<900:
                    header = overlayList[3]
                    drawColor=(0,255,255)
                elif 1050<x1<1200:
                    header = overlayList[4]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drawColor,cv2.FILLED)

            header = cv2.resize(header, (1280, 720))
    #5.if drawing mode - index finger up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("drawing mode")
            if xp ==0 and yp ==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp = x1,y1



    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)   
    _, imgInv= cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)





    # Resize camera feed to the same width as the header (1280px) and reduce height by 100px from the top
    img_resized = cv2.resize(img, (1280, 620))  # New size for the camera feed (width=1280, height=620)

    # Define the region where the camera feed should be placed in the header
    header[100:720, 0:1280] = img_resized  # Place the camera feed starting 100px from the top
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    # Display the final image
    cv2.imshow("Image", header)
    cv2.imshow("Canvas", imgCanvas)
    
    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()