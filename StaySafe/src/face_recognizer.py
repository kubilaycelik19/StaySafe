import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
from settings.settings import CAMERA, FACE_DETECTION, PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        self.names = {}
        self.cam = None
        self.load_model()
        self.load_names()
        self.initialize_camera()
    
    def load_model(self):
        """Load the trained face recognition model"""
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        self.recognizer.read(PATHS['trainer_file'])
        logger.info("Model loaded successfully")
    
    def load_names(self):
        """Load name mappings from JSON file"""
        try:
            if os.path.exists(PATHS['names_file']):
                with open(PATHS['names_file'], 'r') as fs:
                    content = fs.read().strip()
                    if content:
                        self.names = json.loads(content)
        except Exception as e:
            logger.error(f"Error loading names: {e}")
            self.names = {}
    
    def initialize_camera(self):
        """Initialize the camera with error handling"""
        try:
            self.cam = cv2.VideoCapture(CAMERA['index'])
            if not self.cam.isOpened():
                logger.error("Could not open webcam")
                self.cam = None
                return
            
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.cam = None
    
    def recognize_faces(self):
        """Start face recognition loop"""
        if self.cam is None:
            raise ValueError("Failed to initialize camera")
        if self.face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
        
        logger.info("Press 'CTRL + C' to exit.")
        
        while True:
            ret, img = self.cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue

            img = cv2.flip(img, 1)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                name = self.names.get(str(id), "Unknown")
                confidence_text = f"{confidence:.1f}%"
                
                cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cam.release()
        cv2.destroyAllWindows()
        logger.info("Face recognition system stopped.")
        return name

if __name__ == "__main__":
    try:
        face_recognition = FaceRecognitionSystem()
        face_recognition.recognize_faces()
    except Exception as e:
        logger.error(f"System error: {e}")
