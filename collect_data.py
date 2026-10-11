import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

label = "A"

with open("sign_dataset.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()

        # IMPORTANT FIX
        if not ret:
            print("Failed to capture frame")
            continue

        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                landmarks = []

                for lm in hand.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                landmarks.append(label)

                writer.writerow(landmarks)

        cv2.putText(frame, "Collecting: " + label,
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()