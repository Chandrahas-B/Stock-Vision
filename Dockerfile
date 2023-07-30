FROM python:3.9.17-slim-bullseye
COPY . stockVision/
WORKDIR stockVision
RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

CMD streamlit run final.py
EXPOSE 8501
