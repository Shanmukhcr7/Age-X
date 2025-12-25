from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import time
import uvicorn
import logging
from age_service import AgeService

# Setup Logging
logger = logging.getLogger("uvicorn")

# Initialize App
app = FastAPI(
    title="Age-X Safety API",
    description="High-performance age detection for secure content delivery.",
    version="1.0.0"
)

# Security & CORS
# In production, specific origins should be allow-listed.
ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://your-deployment-url.com"  # Replace with actual
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev/demo, strict in prod
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Service (Singleton)
age_service = AgeService()

class ImagePayload(BaseModel):
    image: str  # Base64 encoded string

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
    return response

@app.get("/")
def health_check():
    return {"status": "active", "service": "Age-X API"}

@app.post("/api/age-check")
async def age_check(payload: ImagePayload):
    try:
        # 1. Validation & Decoding
        if not payload.image:
            raise HTTPException(status_code=400, detail="Empty image payload")
        
        try:
            image_bytes = base64.b64decode(payload.image)
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(pil_image)
        except (ValueError, UnidentifiedImageError):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # 2. Inference
        result = age_service.detect_and_predict(image_np)
        
        # 3. Handle Errors (No face, etc)
        if hasattr(result, "get") and result.get("error"):
            # Return "Kid" mode if face not found/error (Fail-Safe)
            return {
                "age_group": "Kid", 
                "confidence": 0.0, 
                "forced_safety": True,
                "msg": result.get("error")
            }

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}")
        # FAIL SAFE: Always return Kid on critical error
        return {
            "age_group": "Kid", 
            "confidence": 0.0, 
            "forced_safety": True,
            "error": "Internal Processing Error"
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
