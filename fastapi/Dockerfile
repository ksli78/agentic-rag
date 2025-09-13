# ===== Stage 1: vendor wheels (online only here) =====
FROM python:3.11-slim AS wheels
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
WORKDIR /vendor
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /vendor/requirements.txt

# 1) Download CUDA Torch stack (cu121)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip download --dest /vendor/wheels --index-url ${TORCH_INDEX_URL} \
      torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# 2) Download ALL app deps (+ transitive) from PyPI
#    (This now pulls pydantic, pydantic-core, annotated-types, typing-inspection, etc.)
RUN pip download --dest /vendor/wheels -r /vendor/requirements.txt

# ===== Stage 2: final runtime (offline) =====
FROM python:3.11-slim
WORKDIR /app
COPY --from=wheels /vendor/wheels /wheels
COPY requirements.txt /app/requirements.txt

# Install CUDA Torch stack offline
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-index --find-links /wheels \
      torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# Install the rest offline (Marker/HF/Starlette/FastAPI etc.)
RUN pip install --no-index --find-links /wheels -r /app/requirements.txt

# Build-time NMS sanity check: fail image if nms missing
RUN python - <<'PY'
import torch, torchvision
from torchvision.ops import nms
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "| TV:", torchvision.__version__)
PY

# App code
COPY . /app/

# Optional: warm marker models for fully offline prod
RUN python - <<'PY'
from marker.models import create_model_dict
create_model_dict(device="cuda")
print("Marker models cached (cuda)")
PY

EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
