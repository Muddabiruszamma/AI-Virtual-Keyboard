import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from time import sleep

# Initialize HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""

# Open webcam (you can also use a video file or other input source)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
cap.set(3, 1280)
cap.set(4, 720)

# Define the size of each key and the spacing between keys
key_width, key_height = 85, 85
spacing = 10

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img


class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []

# Calculate the starting x and y coordinates to center the keyboard on the screen
start_x = (1280 - (len(keys[0]) * key_width + (len(keys[0]) - 1) * spacing)) // 2
start_y = (720 - (len(keys) * key_height + (len(keys) - 1) * spacing)) // 2

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        x = start_x + j * (key_width + spacing)
        y = start_y + i * (key_height + spacing)
        buttonList.append(Button([x, y], key))

# Add Delete button
delete_x = start_x + 9 * (key_width + spacing)
delete_y = start_y + 3 * (key_height + spacing)
buttonList.append(Button([delete_x, delete_y], "Delete", [key_width, key_height]))

lmList = []

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        break

    # Find hands in the frame
    hands, img = detector.findHands(img)
    img = drawAll(img, buttonList)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if lmList:
            # Calculate the center of landmarks 8 and 4
            x1, y1 = lmList[8][0], lmList[8][1]
            x2, y2 = lmList[4][0], lmList[4][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < cx < x + w and y < cy < y + h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 60),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                    distance, _, _ = detector.findDistance([x1, y1], [x2, y2], img)
                    if distance < 25:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 60),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                        if button.text == "Delete":
                            finalText = finalText[:-1]
                        else:
                            finalText += button.text
                        sleep(0.15)

    cv2.rectangle(img, (50, 50), (700, 150), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 120),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    # Display the output
    cv2.imshow("Hand Detection", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
