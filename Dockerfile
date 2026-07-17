FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD env -u STREAMLIT_SERVER_PORT streamlit run app/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
