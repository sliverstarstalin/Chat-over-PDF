FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py"]
