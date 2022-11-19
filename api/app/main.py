from fastapi import FastAPI, Request, HTTPException
from app.request_preprocess import preprocess_text, EmptyText
from app.logging_utils import setup_logger
from app.model_state import ModelState

# app
app = FastAPI()
logger = setup_logger(__name__)
state = ModelState()
classes = ["non-italian", "italian"]


@app.post("/predict")
async def predict(payload: Request):
    data = await payload.json()
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

    prediction = state.predict(text=text)
    prediction_response = {
        "prediction": int(prediction),
        "class": classes[int(prediction)],
    }
    logger.info(prediction_response)
    return prediction_response


@app.get("/")
def home():
    logger.info("Request accepted")
    return {"model": "LanguageClassifier"}
