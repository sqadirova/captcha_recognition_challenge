# CAPTCHA Recognition Challenge

This project is a solution to the CAPTCHA Recognition Challenge Internship Assignment.

## Setup instructions

### 1. Clone the repository from github: 

` git clone https://github.com/sqadirova/captcha_recognition_challenge.git `

` cd captcha_recognision_challenge `

### 2. Build & Run Docker
` docker-compose up --build `

Once the service is running, the API will be accessible at:

` http://localhost:8000/docs `

You can use Swagger UI to test the API easily.

### 3. Configurable Parameters

The following parameters are defined using environment variables via `docker-compose.yml`:

- `MODEL_PATH`: Path to the saved PyTorch model
- `IDX_TO_CHAR_PATH`: Path to the character mapping JSON
- `IMG_WIDTH`, `IMG_HEIGHT`: Input image dimensions
- `DROPOUT_RATE`: Dropout rate in model

This allows flexible deployment without changing the code.

## 4. Example API Call 1 (Swagger)

1. Open: http://localhost:8000/docs

2. Go to POST /predict

3. Click Try it out

4. Upload a CAPTCHA image (.png, .jpg, .jpeg)

5. Click Execute

6. You will receive a response like:

``` 
{
  "prediction": {
    "text": "43C9",
    "confidence": 91.73
  }
}
```

## 5. Example API Call 2 (using curl)

You can send a prediction request using curl:

``` 
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@43C9.png"
```

Expected Response:
```
{
  "prediction": {
    "text": "43C9",
    "confidence": 91.73
  }
}
```

## 6. Assumptions

Model is trained on alphanumeric CAPTCHA images.

Input images will be resized and preprocessed as in the training pipeline.

API accepts only `.png, .jpg, .jpeg` formats.


## 7. Summary

Throughout this challenge, I gained hands-on experience in deep learning model optimization. I explored new concepts such as data augmentation and confidence score generation, which significantly improved my understanding of model robustness and usability. Implementing optional features like rate limiting and confidence scoring helped me enhance the overall API functionality. One key challenge I faced was maintaining consistent accuracy due to randomness in training and augmentation, which taught me the importance of reproducibility and tuning.


## Part 1: Model Improvement

The base model initially had an accuracy of **~40-60%**.  When I run it first it gives **57.5%** accuracy.
I improved the model by applying the following:
- Data Augmentation (Random rotation, affine transformations, color jitter)
- Model Architecture Extension (Added extra CNN and BatchNorm layers)

**Accuracy after improvement:**
Accuracy varied between **70% - 81.25%** due to random seed and augmentation. <br>
Typical accuracy observed: **72%-78.5%**

**Before/After Comparison:**

| Model           | Accuracy       |
|-----------------|:--------------:|
| Base Model     | ~40 - 60%       |
| Improved Model | ~70% - 78.5%    |

**Example Visualization (True vs Predicted):**

| True Label | Predicted | Confidence |
|:---------:|:--------:|:---------:|
| `43C9`   | `43C9`   | 91.73 %  |


## Part 2: API and Containerization

The improved model is deployed as a containerized FastAPI service with the following features:

- **POST /predict** endpoint to accept image file
- Loads and uses the previously saved improved model (from Part 1)
- Returns **predicted CAPTCHA text** and **confidence score**
- Basic error handling and input validation
- Auto-generated API documentation using Swagger
- **Rate limiting** enabled (5 requests per minute)

### Bonus Feature:
- Added **Confidence score** in prediction response


