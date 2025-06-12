import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Key, Controller, KeyCode
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time


class GameController:
    def __init__(self):
        print("Initializing GameController...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.keyboard = Controller()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Initialize control states
        self.current_action = None
        self.steering_angle = 0
        self.active_keys = set()
        self.brake_active = False
        self.hard_brake_active = False

        # Steering parameters
        self.slight_turn_threshold = 30
        self.full_turn_threshold = 45
        self.neutral_zone = 10
        self.prev_steering = None
        self.steering_smoothing = 0.5

    def get_hand_zone(self, hand_landmarks, image_height, image_width):
        """
        Determine which zone the hand is in based on its position
        """
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

        # Draw vertical line in left half
        cv2.line(image, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
        # Draw horizontal line
        cv2.line(image, (0, mid_y), (w // 2, mid_y), (0, 255, 0), 2)

        # Add zone labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Forward", (10, mid_y // 2), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Brake", (mid_x + 10, mid_y // 2), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Backward", (10, mid_y + h // 4), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Hard Brake", (mid_x + 10, mid_y + h // 4), font, 0.8, (0, 255, 0), 2)

    def cleanup_keys(self):
        """Safely release all pressed keys"""
        try:
            keys_to_release = self.active_keys.copy()
            for key in keys_to_release:
                try:
                    self.keyboard.release(key)
                except Exception as e:
                    print(f"Error releasing key {key}: {str(e)}")
            self.active_keys.clear()
        except Exception as e:
            print(f"Error during key cleanup: {str(e)}")

    def press_key(self, key):

        try:
            self.keyboard.press(key)
            self.active_keys.add(key)
        except Exception as e:
            print(f"Error pressing key {key}: {str(e)}")

    def apply_controls(self, gesture_action=None, steering_intensity=None):
        # Clear previous keys
        self.cleanup_keys()

        try:
            # Apply gesture controls
            if gesture_action:
                if gesture_action == 'forward' and not self.brake_active and not self.hard_brake_active:
                    self.press_key(KeyCode.from_char('w'))
                elif gesture_action == 'backward' and not self.brake_active and not self.hard_brake_active:
                    self.press_key(KeyCode.from_char('s'))
                elif gesture_action == 'brake':
                    self.brake_active = True
                    self.hard_brake_active = False
                    self.press_key(Key.space)
                elif gesture_action == 'hard_brake':
                    self.brake_active = False
                    self.hard_brake_active = True
                    self.press_key(KeyCode.from_char('w'))
                    self.press_key(KeyCode.from_char('s'))
                else:
                    self.brake_active = False
                    self.hard_brake_active = False

            # Apply steering controls if not braking
            if steering_intensity is not None and not self.brake_active and not self.hard_brake_active:
                if abs(steering_intensity) == 0.5:  # Slight turn
                    if steering_intensity > 0:
                        self.press_key(KeyCode.from_char('d'))
                    else:
                        self.press_key(KeyCode.from_char('a'))
                elif abs(steering_intensity) == 1.0:  # Full turn
                    if steering_intensity > 0:
                        self.press_key(KeyCode.from_char('d'))
                    else:
                        self.press_key(KeyCode.from_char('a'))

        except Exception as e:
            print(f"Error in apply_controls: {str(e)}")
            self.cleanup_keys()

    def load_and_train(self):
        print("Loading training data...")
        try:
            df = pd.read_csv('control_data/gesture_data.csv')
            X = df.drop('label', axis=1)
            y = df['label']

            print("Training model...")
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            print("Training complete")
            return True
        except FileNotFoundError:
            print("Error: control_data/gesture_data.csv not found!")
            return False
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def run(self):
        print("Starting run sequence...")
        if not self.load_and_train():
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return

        print("\nStarting game control (press 'q' to quit)")
        print("Use LEFT HALF zones for gestures:")
        print(" - Upper Left:  Forward")
        print(" - Upper Right: Normal Brake (spacebar)")
        print(" - Lower Left:  Backward")
        print(" - Lower Right: Hard Brake (W+S)")
        print("\nUse RIGHT HALF for steering controls:")
        print(f" - Neutral zone: ±{self.neutral_zone}°")
        print(f" - Slight turn: ±{self.slight_turn_threshold}°")
        print(f" - Full turn: ±{self.full_turn_threshold}°")

        try:
            while True:
                success, image = cap.read()
                if not success:
                    continue

                image = cv2.flip(image, 1)
                h, w = image.shape[:2]

                # Draw zone divisions and labels
                self.draw_zones(image)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                current_gesture = None
                current_steering = None

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Check if hand is in left half
                        hand_zone = self.get_hand_zone(hand_landmarks, h, w)

                        if hand_zone:
                            # Process gestures in left half
                            features = []
                            for landmark in hand_landmarks.landmark:
                                features.extend([landmark.x, landmark.y])

                            features_scaled = self.scaler.transform([features])
                            current_gesture = self.model.predict(features_scaled)[0]

                            # Display current action
                            cv2.putText(image, f"Action: {current_gesture}", (10, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            # Process steering in right half
                            angle = self.calculate_tilt_angle(hand_landmarks)
                            steering_intensity = self.get_steering_intensity(angle)
                            current_steering = steering_intensity

                            # Display steering status
                            if abs(angle) < self.neutral_zone:
                                status = "Neutral"
                            else:
                                direction = "Right" if angle > 0 else "Left"
                                intensity = "Full" if abs(steering_intensity) == 1.0 else "Slight"
                                status = f"{intensity} {direction} ({abs(angle):.1f}°)"

                            cv2.putText(image, f"Steering: {status}", (w // 2 + 10, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Apply both controls after processing all hands
                    self.apply_controls(gesture_action=current_gesture, steering_intensity=current_steering)

                cv2.imshow('Game Control', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup_keys()
            cap.release()
            cv2.destroyAllWindows()

    def calculate_tilt_angle(self, hand_landmarks):
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y

        angle = np.degrees(np.arctan2(dx, -dy))

        if self.prev_steering is not None:
            angle = (self.steering_smoothing * self.prev_steering +
                     (1 - self.steering_smoothing) * angle)
        self.prev_steering = angle

        return angle

    def get_steering_intensity(self, angle):
        if abs(angle) < self.neutral_zone:
            return 0
        if abs(angle) < self.slight_turn_threshold:
            return 0.5 if angle > 0 else -0.5
        return 1.0 if angle > 0 else -1.0

    def __del__(self):
        """Destructor to ensure all keys are released"""
        self.cleanup_keys()


if __name__ == "__main__":
    try:
        controller = GameController()
        controller.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        input("Press Enter to exit...")
    finally:
        if 'controller' in locals():
            controller.cleanup_keys()