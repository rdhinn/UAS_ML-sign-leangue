FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    mv /usr/local/bin/streamlit /usr/local/bin/streamlit.real && \
    printf '#!/bin/sh\nunset STREAMLIT_SERVER_PORT\nexec /usr/local/bin/streamlit.real "$@"\n' > /usr/local/bin/streamlit && \
    chmod +x /usr/local/bin/streamlit

COPY . .

EXPOSE 8501

CMD streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
