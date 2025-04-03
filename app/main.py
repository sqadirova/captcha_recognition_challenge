# from fastapi import FastAPI, UploadFile, File, HTTPException
# from app.model import predict_captcha
# from PIL import Image
# from io import BytesIO

# app = FastAPI(title="CAPTCHA Recognition API", version="1.0")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         raise HTTPException(status_code=400, detail="Invalid file format")

#     try:
#         image_bytes = await file.read()
#         image = Image.open(BytesIO(image_bytes)).convert("L")
#         prediction = predict_captcha(image)
#         return {"prediction": prediction}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.model import predict_captcha
from io import BytesIO
from PIL import Image

app = FastAPI(title="CAPTCHA Recognition API", version="1.0")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("L")
        prediction = predict_captcha(image)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

