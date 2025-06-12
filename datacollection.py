import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os


class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_features(self, hand_landmarks):
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y])
        return features

    def get_hand_zone(self, hand_landmarks, image_height, image_width):
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        x, y = wrist.x * image_width, wrist.y * image_height

        if x >= image_width / 2:
            return None

        mid_x = image_width / 4
        mid_y = image_height / 2

        if y < mid_y:  # Upper half
            if x < mid_x:
                return "forward_zone"
            else:
                return "brake_zone"
        else:  # Lower half
            if x < mid_x:
                return "backward_zone"
            else:
                return "hard_brake_zone"

    def draw_zones(self, image):
        h, w = image.shape[:2]
        mid_x = w // 4
        mid_y = h // 2

        # Draw zones
        cv2.line(image, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
        cv2.line(image, (0, mid_y), (w // 2, mid_y), (0, 255, 0), 2)
        cv2.line(image, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)

        # Add zone labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Forward", (10, mid_y // 2), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Brake", (mid_x + 10, mid_y // 2), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Backward", (10, mid_y + h // 4), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Hard Brake", (mid_x + 10, mid_y + h // 4), font, 0.8, (0, 255, 0), 2)

    def collect_data(self):
        training_data = []
        labels = []

        # Initialize camera with DirectShow backend
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\nCollecting training data for hand gestures...")
        print("Use LEFT HALF of screen divided into 4 zones:")
        print("Upper Left  (1) - Forward  (press 'w')")
        print("Upper Right (2) - Brake    (press 'b')")
        print("Lower Left  (3) - Backward (press 's')")
        print("Lower Right (4) - Hard Brake (press 'h')")
        print("Press 'q' to quit")

        while True:
            ret, image = cap.read()
            if not ret:
                continue

            image = cv2.flip(image, 1)
            h, w = image.shape[:2]

            self.draw_zones(image)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    hand_zone = self.get_hand_zone(hand_landmarks, h, w)

                    if hand_zone:
                        cv2.putText(image, f"Zone: {hand_zone}", (10, h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, f"Samples: {len(training_data)}", (10, h - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Data Collection - Hand Gestures', image)
            key = cv2.waitKey(1) & 0xFF

            if results.multi_hand_landmarks and key in [ord('w'), ord('s'), ord('b'), ord('h')]:
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_zone = self.get_hand_zone(hand_landmarks, h, w)

                if hand_zone:
                    features = self.extract_features(hand_landmarks)
                    training_data.append(features)

                    if key == ord('w') and hand_zone == "forward_zone":
                        labels.append('forward')
                    elif key == ord('s') and hand_zone == "backward_zone":
                        labels.append('backward')
                    elif key == ord('b') and hand_zone == "brake_zone":
                        labels.append('brake')
                    elif key == ord('h') and hand_zone == "hard_brake_zone":
                        labels.append('hard_brake')
                    else:
                        training_data.pop()
                        print("Hand not in correct zone for this action")
                        continue

                    print(f"Collected {chr(key)} in {hand_zone} - Total: {len(training_data)}")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if training_data:
            self.save_data(training_data, labels)

    def save_data(self, training_data, labels):
        df = pd.DataFrame(training_data)
        df['label'] = labels
        os.makedirs('control_data', exist_ok=True)
        df.to_csv('control_data/gesture_data.csv', index=False)
        print("\nData saved to control_data/gesture_data.csv")


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()