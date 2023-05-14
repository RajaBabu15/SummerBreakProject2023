# SignLanguageDetectionApp

SignLanguageDetectionApp is a Python application that uses LSTM sequences to detect and translate sign language gestures from a video stream. It can recognize signs from the American Sign Language (ASL) alphabet and some common words and phrases.

## Features

- Real-time sign language detection and translation
- Continuous gesture recognition without frame-by-frame segmentation
- Support for both webcam and video file input
- User-friendly graphical user interface (GUI)

## Requirements

- Python 3.7 or higher
- OpenCV 4.5 or higher
- TensorFlow 2.4 or higher
- Keras 2.4 or higher
- NumPy 1.19 or higher
- PyQt5 5.15 or higher

## Installation

To install the SignLanguageDetectionApp, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory and create a virtual environment using `python -m venv env`.
3. Activate the virtual environment using `env\Scripts\activate` on Windows or `source env/bin/activate` on Linux/MacOS.
4. Install the required packages using `pip install -r requirements.txt`.
5. Run the main script using `python main.py`.

## Usage

To use the SignLanguageDetectionApp, follow these steps:

1. Run the main script using `python main.py`.
2. Select the video source from the drop-down menu. You can choose either your webcam or a video file from your computer.
3. Click the "Start" button to start the sign language detection and translation.
4. Perform sign language gestures in front of the camera or play the video file. The app will display the detected signs and their meanings on the screen.
5. Click the "Stop" button to stop the sign language detection and translation.
6. Click the "Exit" button to close the app.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project is based on the following papers and repositories:

- [Real-Time Sign Language Recognition using Human Pose Estimation](https://arxiv.org/abs/1809.11096)
- [Sign Language Recognition using Temporal Classification](https://arxiv.org/abs/1702.04567)
- [Sign Language Recognition with LSTM](https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn)