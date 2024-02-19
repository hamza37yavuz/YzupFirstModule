FROM python:3.12
WORKDIR /app
COPY . .
RUN apt-get update
RUN pip install -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py"]