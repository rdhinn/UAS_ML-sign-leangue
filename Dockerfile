FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f /usr/local/bin/streamlit* && \
    printf '#!/usr/bin/env python3\nimport os,sys\nos.environ.pop("STREAMLIT_SERVER_PORT",None)\nsys.argv[0]="streamlit"\nfrom streamlit.web.cli import main\nmain()\n' > /usr/local/bin/streamlit && \
    chmod +x /usr/local/bin/streamlit

COPY . .

EXPOSE 8501

CMD streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
