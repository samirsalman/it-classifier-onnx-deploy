from fastapi import FastAPI, Request, HTTPException
from app.request_preprocess import preprocess_text, EmptyText
from app.logging_utils import setup_logger

# app
app = FastAPI()
logger = setup_logger(__name__)


@app.post("/predict")
async def predict(payload: Request):
    data = payload.json()
    if "text" not in data:
        logger.error(
            "Request refused, error 400: text field is missing in the request body"
        )
        raise HTTPException(
            status_code=400, detail="text field is missing in the request body"
        )

    logger.info("Request arrived")
    text = data["text"]
    try:
        text = preprocess_text(text=text)
    except EmptyText as e:
        logger.error(f"Request refused, error 400: {e}")
        raise HTTPException(status_code=400, detail=e)
    # prediction(text)
    prediction_response = {"prediction": "", "confidence": "", "class": ""}
    logger.info(prediction_response)
    return prediction_response


@app.post("/predict_batch")
async def predict(payload: Request):
    data = payload.json()
    if "texts" not in data or not isinstance(data["texts"], list):
        logger.error(
            "Request refused, error 400: text field is missing in the request body or is not a list"
        )

        raise HTTPException(
            status_code=400,
            detail="text field is missing in the request body or is not a list",
        )
    batch = data["texts"]
    for index, text in batch:
        batch[index] = preprocess_text(text=text)
    # prediction(batch)
    prediction_response = {"prediction": "", "confidence": "", "class": ""}
    return prediction_response


@app.get("/")
def home():
    logger.info("Request accepted")
    return {"model": "LanguageClassifier"}
