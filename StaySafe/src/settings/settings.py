"""
Configuration settings for the face recognition system
"""
import os

# Get the root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Camera settings
CAMERA = {
    'index': 0,  # Default camera (0 is usually built-in webcam)
    'width': 640,
    'height': 480
}

# Face detection settings
FACE_DETECTION = {
    'scale_factor': 1.3,
    'min_neighbors': 5,
    'min_size': (30, 30)
}

# Training settings. Number of images needed to train the model.
TRAINING = {
    'samples_needed': 120
}

# training dependencies
PATHS = {
    'image_dir': os.path.join(ROOT_DIR, 'images'),
    'cascade_file': os.path.join(ROOT_DIR, 'haarcascade_frontalface_default.xml'),
    'names_file': os.path.join(ROOT_DIR, 'names.json'),
    'trainer_file': os.path.join(ROOT_DIR, 'trainer.yml')
}
