FROM python:3.12-slim
WORKDIR /dashboard
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit","run","home.py","--server.address=0.0.0.0","--server.port=8501"]