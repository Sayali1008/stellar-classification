"""HTTP API for the classifier: the demo page, /predict, /health, and the automatic /docs page.

Runs locally with uvicorn (`uvicorn app.main:app --reload`) and on AWS Lambda through the
Mangum adapter (`handler` at the bottom). Imports only serving code, never the training modules.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from mangum import Mangum

from src.predict import load_model, model_version, predict
from src.schemas import PredictRequest, PredictResponse

INDEX_PAGE = Path(__file__).parent / 'static' / 'index.html'

# Load the model at start-up, not on the first request. On Lambda this runs in the init phase,
# and a missing or incompatible model stops the app here instead of failing someone's request.
load_model()

app = FastAPI(
    title='Stellar classification',
    description='Classifies SDSS objects as GALAXY, STAR or QSO from their u, g, r, i, z magnitudes.',
)


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(INDEX_PAGE)


@app.get('/health')
def health():
    """The app is up, and which model it is serving."""
    return {'status': 'ok', 'model_version': model_version()}


@app.post('/predict', response_model=PredictResponse)
def predict_classes(request: PredictRequest):
    """Classify 1 to 1000 objects. Invalid input never gets here: FastAPI answers 422 with the reason."""
    return PredictResponse(model_version=model_version(), predictions=predict(request.objects))


# Lambda entry point: turns a Lambda event into an HTTP request for the app, and the response back.
# lifespan='off' because the app has no start-up/shut-down hooks (the model loads at import above).
handler = Mangum(app, lifespan='off')
