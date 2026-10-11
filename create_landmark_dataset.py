import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = "asl_dataset"

data = []

for label in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    print("Processing:", label)

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        # FIX: resize image so mediapipe detects hand
        image = cv2.resize(image, (640, 480))

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                landmarks = []

                for lm in hand.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                landmarks.append(label)

                data.append(landmarks)

print("Total samples collected:", len(data))

with open("sign_dataset.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerows(data)

print("Dataset Created Successfully!")