#!/usr/bin/env python3
import os

os.environ.pop("STREAMLIT_SERVER_PORT", None)

try:
    port = int(os.environ.get("PORT", 8501))
except (ValueError, TypeError):
    port = 8501

os.environ["STREAMLIT_SERVER_PORT"] = str(port)

from streamlit.web import bootstrap

if __name__ == "__main__":
    bootstrap.run("app/app.py", "", [], flag_options={
        "server.port": port,
        "server.address": "0.0.0.0",
    })
