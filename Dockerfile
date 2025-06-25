FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/model.h5 /app/models/model.h5

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
CMD ["python", "app.py"]  

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]