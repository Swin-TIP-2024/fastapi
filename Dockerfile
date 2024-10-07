FROM python:3.9-slim

# Install JRE
RUN apt-get update && apt-get install -y default-jre

# Set working directory
WORKDIR /app

# Copy only necessary files/folders
COPY requirements.txt .
COPY main.py .
COPY assets/ assets/
COPY models/ models/
COPY utils/ utils/

RUN pip install -r requirements.txt
RUN chmod 777 assets/decompile_apk.sh
RUN chmod 777 assets/apktool_2.10.0.jar

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]