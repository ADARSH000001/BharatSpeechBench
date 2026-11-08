# CLI image for reproducible benchmark runs (bsb run / report). The dashboard is run locally: streamlit run app/Home.py
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[whisper]"
COPY . .
ENTRYPOINT ["bsb"]
