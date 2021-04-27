FROM python 3.8
RUN pip install --upgrade pip
COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD uvicorn deploy:app --reload