FROM python:3.9-slim

WORKDIR /app

COPY ./app ./app
COPY ./model ./model

RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
