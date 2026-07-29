from python:3.10-slim

# working directory
WORKDIR /app

# requirements file ko container me copy
COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# command for start server
CMD ["uvicorn", "AQI_fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]




