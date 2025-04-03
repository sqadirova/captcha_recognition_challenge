from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from io import BytesIO
from app.model import predict_captcha
from PIL import Image


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CAPTCHA Recognition API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(status_code=429, content={"error": "Too many requests"}))

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(request: Request, file: UploadFile = File(..., description="Upload .png, .jpg, .jpeg image file")):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("L")
        prediction = predict_captcha(image)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

