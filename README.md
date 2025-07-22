# Virtual Driving with Hand Gestures

This project allows you to control driving games or other desktop applications using real-time hand gestures captured through your webcam. It uses computer vision to track your hands, recognize specific gestures, and translate them into keyboard commands.

## Features

* **Real-time Hand Tracking**: Utilizes Google's MediaPipe library for fast and accurate hand landmark detection.
* **Gesture Recognition**: Employs a Scikit-learn RandomForestClassifier to distinguish between different hand gestures (e.g., forward, brake).
* **Steering Control**: Detects the tilt angle of the right hand to simulate steering left and right.
* **Keyboard Simulation**: Uses `pynput` to press keyboard keys (`W`, `A`, `S`, `D`, `Spacebar`) that control the game.
* **Data Collection Script**: Includes a separate script (`datacollection.py`) to easily capture and label your own gesture data.

## How It Works

The project is split into two main parts:

1.  **Data Collection (`datacollection.py`)**: You first run this script to capture data. You place your hand in different zones on the screen and press the corresponding key (`w` for forward, `b` for brake, etc.) to label the gesture. This data is saved to a `gesture_data.csv` file.
2.  **Live Control (`testing.py`)**: This script loads the collected data, trains a machine learning model, and then starts the webcam. It analyzes your hand movements in real-time, predicts your intended action, and presses the corresponding keys on your keyboard to control a game.

## Core Technologies

* **Python 3**
* **OpenCV**: For webcam access and image processing.
* **MediaPipe**: For hand landmark detection.
* **Scikit-learn**: For training the gesture recognition model.
* **Pandas**: For handling the training data.
* **pynput**: For controlling the keyboard.

## Setup and Usage

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
