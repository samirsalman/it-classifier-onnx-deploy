FROM python:3.9.15-slim

WORKDIR /api
COPY api/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cpu
COPY ./api/app /api/app
COPY ./runtime /api/app/runtime
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]