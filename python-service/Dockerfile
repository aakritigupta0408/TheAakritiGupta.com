FROM python:3.11-slim

# System deps: gcc for pyswisseph C extension, git for pip install
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the vedic-astro-ai package directly from GitHub.
# The repo includes the ephe/ Swiss Ephemeris files, so no separate download needed.
RUN pip install --no-cache-dir \
    "vedic-astro-ai @ git+https://github.com/aakritigupta0408/vedic-astro-ai.git"

# Pull the gradio_app entrypoint and ephe data from the same repo.
# We clone separately (after pip install) so the working tree is available.
RUN git clone --depth 1 https://github.com/aakritigupta0408/vedic-astro-ai.git /app/src

# Point the settings at the bundled ephemeris path
ENV SWISSEPH_PATH=/app/src/ephe

# Render assigns PORT at runtime; Gradio reads it via os.environ.get("PORT", 7860)
WORKDIR /app/src

CMD ["python", "ui/gradio_app.py"]
