import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    for id in range(0, 5):
        x, y = hand_landmarks.landmark[tip_ids[id]].x, hand_landmarks.landmark[tip_ids[id]].y
        if id == 0:
            if x < hand_landmarks.landmark[tip_ids[id] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

    return fingers.count(1)

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_side = ""
                    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x:
                        hand_side = "Right"
                    else:
                        hand_side = "Left"

                    finger_count = count_fingers(hand_landmarks)

                    if hand_side == "Left":
                        if finger_count == 1:
                            print("w")
                        elif finger_count == 4:
                            print("wa")
                        elif finger_count == 5:
                            print("stop")
                    elif hand_side == "Right":
                        if finger_count == 1:
                            print("s")
                        elif finger_count == 4:
                            print("wd")
                        elif finger_count == 5:
                            print("stop")

            cv2.imshow('Hand Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()