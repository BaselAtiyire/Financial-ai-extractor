FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY api.py app.py ./

# Copy start script
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
