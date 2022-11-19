FROM python:3.9.15-slim

WORKDIR /api
COPY requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt
COPY ./app /api/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]