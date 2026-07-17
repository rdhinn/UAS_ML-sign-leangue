FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD python -c "import os; os.environ.pop('STREAMLIT_SERVER_PORT',None); os.execvp('streamlit',['streamlit','run','app/app.py','--server.port=8501','--server.address=0.0.0.0'])"
