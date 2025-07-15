import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

clicked = False
dragging = False
drag_start_time = None

scrolling = False
scroll_start_time = None

screen_w, screen_h = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Índice y pulgar para mover/click/drag
            x1 = hand_landmarks.landmark[8].x
            y1 = hand_landmarks.landmark[8].y
            x2 = hand_landmarks.landmark[4].x
            y2 = hand_landmarks.landmark[4].y
            cx1, cy1 = int(x1 * w), int(y1 * h)
            cx2, cy2 = int(x2 * w), int(y2 * h)
            distance = math.hypot(cx2 - cx1, cy2 - cy1)
            cx_mid, cy_mid = (cx1 + cx2) // 2, (cy1 + cy2) // 2

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            screen_x = int(cx_mid * screen_w / w)
            screen_y = int(cy_mid * screen_h / h)
            pyautogui.moveTo(screen_x, screen_y, duration=0)

            # Gesto índice-pulgar (drag/click)
            if distance < 25:
                print(f"Moving to: {screen_x}, {screen_y}")
                cv2.circle(frame, (cx_mid, cy_mid), 10, (255, 0, 255), cv2.FILLED)
                if drag_start_time is None:
                    drag_start_time = time.time()
                elif not dragging and time.time() - drag_start_time > 1:
                    pyautogui.mouseDown()
                    dragging = True
                    print("Dragging started")
                elif not clicked and not dragging:
                    pyautogui.click(screen_x, screen_y)
                    clicked = True
                    print("Click registered")
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                    print("Dragging ended")
                drag_start_time = None
                clicked = False


            # Gesto scroll: índice y medio levantados y juntos
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]
            fingers_up = []
            for tip, pip in zip(finger_tips, finger_pips):
                tip_y = hand_landmarks.landmark[tip].y
                pip_y = hand_landmarks.landmark[pip].y
                fingers_up.append(tip_y < pip_y)

            ix, iy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            mx, my = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)
            index_middle_distance = math.hypot(mx - ix, my - iy)

            if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3] and index_middle_distance < 40:
                if scroll_start_time is None:
                    scroll_start_time = time.time()
                elif not scrolling and time.time() - scroll_start_time > 0.1:
                    pyautogui.mouseDown(button='middle')
                    scrolling = True
                    print("Scrolling started")
            else:
                if scrolling:
                    pyautogui.mouseUp(button='middle')
                    scrolling = False
                scroll_start_time = None

    cv2.imshow("Hand Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
