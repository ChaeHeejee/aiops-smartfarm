FROM python:3.11-slim
WORKDIR /app
COPY tom_aiops/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
EXPOSE 8010
CMD ["uvicorn", "tom_aiops.app:app", "--host", "0.0.0.0", "--port", "8010"]
