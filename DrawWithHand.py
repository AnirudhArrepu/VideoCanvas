import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

cap = cv2.VideoCapture(0)
istrue,frame = cap.read()
blank = np.zeros((frame.shape[0],frame.shape[1],3), dtype='uint8')
def DrawOnBoard(coords):
    x = coords[0]
    y = coords[1]
    blank[y][x] = 0,255,0
    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = frame.shape
            thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            distance = calculate_distance(thumb_tip_coords, index_tip_coords)

            threshold = 20 
            if distance < threshold:
                cv2.putText(frame, "Touched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                DrawOnBoard(thumb_tip_coords)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)
    cv2.imshow('DRAW', blank)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
