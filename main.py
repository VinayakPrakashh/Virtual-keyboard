import cv2
from pynput.keyboard import Key,Controller
from cvzone.HandTrackingModule import HandDetector
from time import sleep
cap = cv2.VideoCapture(0) 
cap.set(3, 2400)
cap.set(4, 1440)
finalText = ""
detector = HandDetector(detectionCon=1)
keys = [["Q","W","E","R","T","Y","U","I","Q","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]]
Keyboard = Controller()

def drawAll(img,ButtonList):
        
        for button in ButtonList:
             
            x,y=button.pos
            w,h=button.size
            cv2.rectangle(img,button.pos,(x + w,y + h),(255,0,255),cv2.FILLED)
            cv2.putText(img,button.text,(x+21,y+ 65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
        return img

class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text
        
        

ButtonList = []
for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            ButtonList.append(Button([100 * j + 58,100 * i + 58], key))


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bboxInfo = detector.findPosition(img)
    img = drawAll(img, ButtonList)

    if lmlist:
         for button in ButtonList:
              x,y = button.pos
              w,h = button.size

              if x<lmlist[8][0]<x+w and y<lmlist[8][1]<y+h:
                   cv2.rectangle(img,button.pos,(x + w,y + h),(175,0,175),cv2.FILLED)
                   cv2.putText(img,button.text,(x+21,y+ 65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                   l,_,_ = detector.findDistance(8,12,img,draw=False)
                
                   
                   if l<38:
                         Keyboard.press(button.text)
                         cv2.rectangle(img,button.pos,(x + w,y + h),(0,255,0),cv2.FILLED)
                         cv2.putText(img,button.text,(x+21,y+ 65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                         finalText += button.text
                         sleep(0.15)
    cv2.rectangle(img,(58,358),(788,458),(175,0,175),cv2.FILLED)
    cv2.putText(img,finalText,(68,425),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)