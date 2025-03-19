import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
from . settings.settings import CAMERA, FACE_DETECTION, PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.names = {}
        self.cam = None
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Model yüklenemedi: {e}")
        self.load_names()
        self.initialize_camera()
    
    def load_model(self):
        """Load the trained face recognition model"""
        try:
            trainer_path = PATHS['trainer_file']
            logger.info(f"Trainer dosyası yolu: {trainer_path}")
            logger.info(f"Dosya mevcut mu: {os.path.exists(trainer_path)}")
            if os.path.exists(trainer_path):
                self.recognizer.read(trainer_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning("Trainer file not found. Face recognition will be limited.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def load_names(self):
        """Load name mappings from JSON file"""
        try:
            names_path = PATHS['names_file']
            logger.info(f"Names dosyası yolu: {names_path}")
            logger.info(f"Dosya mevcut mu: {os.path.exists(names_path)}")
            if os.path.exists(names_path):
                with open(names_path, 'r') as fs:
                    content = fs.read().strip()
                    if content:
                        self.names = json.loads(content)
                        logger.info(f"Names yüklendi: {self.names}")
            else:
                logger.warning("Names file not found.")
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
    
    def recognize_faces(self, img):
        """Yüz tanıma işlemi"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # İlk yüzü al
                try:
                    id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    name = self.names.get(str(id), "Unknown")
                except:
                    name = "Unknown"
                    confidence = 0
                return name, confidence
            
            return "Unknown", 0
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return "Error", 0

if __name__ == "__main__":
    try:
        face_recognition = FaceRecognitionSystem()
        face_recognition.recognize_faces()
    except Exception as e:
        logger.error(f"System error: {e}")
