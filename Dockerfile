# =============================================
# FlightInsight - Streamlit Dashboard Dockerfile
# =============================================

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install uv as a single binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies in venv
RUN uv sync --frozen --no-dev

# Add venv to PATH
ENV PATH="/app/.venv/bin:${PATH}"

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/

# Copy models and supporting data
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/reviews/ ./data/reviews/

# Copy sample data for Analytics (NE polni flights.csv, samo vzorec)
COPY data/preprocessed/flights_sample.csv ./data/preprocessed/flights_sample.csv

EXPOSE 8501

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_THEME_BASE=light

CMD ["streamlit", "run", "frontend/Home.py"]