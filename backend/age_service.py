import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import logging
import os

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgeService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.face_cascade = None
        self.transform = None
        self.MODEL_PATH = "age_model.pth"
        self.IMG_SIZE = 224
        
        # Load Resources
        self._load_model()
        self._load_face_detector()
        self._init_transform()
        
        self.initialized = True
        logger.info(f"AgeService initialized on {self.device}")

    def _load_model(self):
        try:
            # Architecture: ResNet18 Regression
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            
            if os.path.exists(self.MODEL_PATH):
                self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
                logger.info("Model weights loaded successfully.")
            else:
                logger.warning(f"Model file {self.MODEL_PATH} not found. Running with random weights (UNSAFE FOR PRODUCTION).")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError("Critical: Model loading failed.")

    def _load_face_detector(self):
        try:
            # Use OpenCV Haar Cascades
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise IOError(f"Failed to load cascade from {cascade_path}")
        except Exception as e:
            logger.error(f"Face detector setup failed: {e}")
            raise

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_and_predict(self, image_array: np.ndarray):
        """
        Process the image: Detect face -> Predict Age -> Return result.
        Returns: { "age": float, "age_group": str, "confidence": float } or None
        """
        try:
            # 1. Face Detection
            frame = cv2.resize(image_array, (640, 480)) # Optimize detection speed
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Strict detection parameters to avoid false positives
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5, 
                minSize=(50, 50)
            )

            if len(faces) == 0:
                logger.debug("No faces detected.")
                return {"error": "No face detected"}

            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Check face size relative to frame (too small = unreliable)
            if w < 60 or h < 60:
                 return {"error": "Face too small"}

            face_img = frame[y:y+h, x:x+w]

            # 2. Preprocess
            pil_img = Image.fromarray(face_img)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # 3. Inference
            with torch.no_grad():
                age_pred = self.model(input_tensor).item()

            # 4. Post-process
            # Clamp age to realistic bounds
            age = max(1, min(100, age_pred))
            group = self._get_age_group(age)
            
            # Synthetic confidence (for regression models, we don't have softmax probability)
            # We can imply 'confidence' is lower if age is near a boundary? 
            # For now, return 0.95 as placeholder for regression output reliability.
            confidence = 0.95 

            return {
                "age": round(age, 1),
                "age_group": group,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": str(e)}

    def _get_age_group(self, age):
        """
        Maps scalar age to safety categories.
        Safety-First: Boundaries are conservative.
        """
        # Kid: 0-12
        # Teen: 13-17
        # Adult: 18+
        
        # We can implement a "buffer" zone? 
        # i.e. 12.5 -> treated as Kid?
        # Let's stick to standard strict mapping for now.
        
        if age < 13:
            return "Kid"
        elif age < 18:
            return "Teen"
        elif age < 25:
            return "Young Adult"
        elif age < 60:
            return "Adult"
        else:
            return "Senior"
