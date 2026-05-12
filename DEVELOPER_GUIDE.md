# AI Service Developer Guide

> Handcrafted Jewelry E-Commerce Platform — Image Feature Extraction Microservice

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Environment Configuration](#environment-configuration)
6. [Running the Service](#running-the-service)
7. [Project Structure](#project-structure)
8. [Architecture](#architecture)
9. [API Reference](#api-reference)
10. [How It Integrates with the Backend](#how-it-integrates-with-the-backend)
11. [Common Development Tasks](#common-development-tasks)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The AI Service is a lightweight Python Flask microservice that extracts **visual feature vectors** from product images using **MobileNetV2** (pre-trained on ImageNet). These 1280-dimensional, L2-normalized embeddings power the platform's visual similarity search feature — customers can upload a photo of jewelry and find visually similar products in the catalog.

**Key responsibilities:**

- Accept an image (file upload or URL) and return a normalized feature vector
- Serve as a stateless HTTP microservice called by the Node.js backend
- Preload the MobileNetV2 model on startup for fast inference

---

## Tech Stack

| Technology | Version | Purpose                               |
| ---------- | ------- | ------------------------------------- |
| Python     | 3.11    | Runtime                               |
| Flask      | 3.0.0   | HTTP microservice framework           |
| Flask-CORS | 4.0.0   | Cross-origin request handling         |
| TensorFlow | 2.15.0  | MobileNetV2 model inference           |
| Pillow     | 10.2.0  | Image loading and preprocessing       |
| NumPy      | 1.26.3  | Array operations and L2 normalization |
| Requests   | 2.31.0  | Fetching images from remote URLs      |

---

## Prerequisites

- **Python 3.11** — [python.org](https://www.python.org/downloads/)
  > TensorFlow 2.15 supports Python 3.8–3.11. **Python 3.12+ is not supported.**
- **pip** (comes with Python)
- A virtual environment tool (`venv` is built into Python)
- ~500 MB disk space for TensorFlow and the MobileNetV2 weights

---

## Installation

### 1. Navigate to the project directory

```bash
cd "d:\Final Projects\E-Commerce Platform\System\ai-service"
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads TensorFlow (~400 MB). This may take a few minutes depending on your connection.

### 4. Verify installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.15.x
```

---

## Environment Configuration

The service is configured via environment variables. You can set them in your shell or create a `.env` file and load it manually.

| Variable                         | Default          | Description                                     |
| -------------------------------- | ---------------- | ----------------------------------------------- |
| `PORT`                           | `5001`           | Port the service listens on                     |
| `AI_SERVICE_MAX_IMAGE_BYTES`     | `8388608` (8 MB) | Maximum accepted image size in bytes            |
| `AI_SERVICE_URL_TIMEOUT_SECONDS` | `20`             | Timeout (seconds) for fetching images from URLs |
| `AI_SERVICE_PRELOAD_MODEL`       | `1`              | Set to `0` to skip model preload on startup     |

### Setting variables on Windows (CMD)

```cmd
set PORT=5001
set AI_SERVICE_PRELOAD_MODEL=0
python app.py
```

### Setting variables on Windows (PowerShell)

```powershell
$env:PORT = "5001"
$env:AI_SERVICE_PRELOAD_MODEL = "0"
python app.py
```

### Setting variables on macOS / Linux

```bash
PORT=5001 AI_SERVICE_PRELOAD_MODEL=0 python app.py
```

---

## Running the Service

Make sure your virtual environment is activated first.

### Standard startup

```bash
python app.py
```

On startup you will see:

```
INFO:root:Loading MobileNetV2 model...
INFO:root:MobileNetV2 model loaded successfully.

==================================================
AI Feature Extraction Service
==================================================
Running on: http://localhost:5001
Model: MobileNetV2 (Transfer Learning)
Feature Vector Size: 1280 dimensions
==================================================
```

### Alternative: Flask CLI

```bash
flask run --port 5001
```

### Fast startup (skip model preload)

```bash
# Windows CMD
set AI_SERVICE_PRELOAD_MODEL=0 & python app.py

# macOS / Linux
AI_SERVICE_PRELOAD_MODEL=0 python app.py
```

The model will load on the **first request** instead (~2–5 seconds delay on first call).

### Custom port

```bash
# Windows CMD
set PORT=5002 & python app.py

# macOS / Linux
PORT=5002 python app.py
```

---

## Project Structure

```
ai-service/
├── app.py              ← Entire Flask application (single file)
├── requirements.txt    ← Python dependencies
├── README.md
└── tests/
    └── test_app.py     ← Unit and integration tests
```

The service is intentionally kept as a single file (`app.py`) for simplicity. All logic — model loading, image preprocessing, feature extraction, and HTTP endpoints — lives in this one file.

---

## Architecture

### Image Processing Pipeline

Every image goes through the same preprocessing pipeline before feature extraction:

```
Input (uploaded file or URL)
        │
        ▼
  Validate size (≤ 8 MB)
        │
        ▼
  Open with Pillow
        │
        ▼
  EXIF transpose
  (fix camera rotation)
        │
        ▼
  Convert to RGB
  (RGBA/LA → white background composite → RGB)
        │
        ▼
  Center-crop & resize to 224×224
  (LANCZOS resampling)
        │
        ▼
  MobileNetV2 preprocess_input
  (scales pixel values to [-1, 1])
        │
        ▼
  Add batch dimension → shape (1, 224, 224, 3)
        │
        ▼
  MobileNetV2.predict()
  → shape (1, 1280)
        │
        ▼
  Flatten → L2 normalize
  → 1280-float list
        │
        ▼
  JSON response
```

### MobileNetV2 Model Configuration

```python
tf.keras.applications.MobileNetV2(
    weights="imagenet",   # Pre-trained on ImageNet
    include_top=False,    # Remove classification head
    pooling="avg",        # Global average pooling → 1280-dim output
)
```

The model is loaded once and cached in the global `_feature_model` variable. Subsequent requests reuse the same model instance.

### L2 Normalization

After extraction, the feature vector is L2-normalized:

```
normalized = features / ||features||₂
```

This ensures all vectors have unit length, making **cosine similarity** equivalent to a simple dot product — which is how the backend compares vectors for search.

### Error Handling

The service uses two custom exception classes:

| Exception                  | When raised                                                                     |
| -------------------------- | ------------------------------------------------------------------------------- |
| `InvalidImageRequestError` | Empty payload, file too large, not a valid image, URL returns non-image content |
| `FeatureExtractionError`   | Model returns a zero-length vector (extremely rare)                             |

HTTP status codes:

- `400` — Invalid request (bad image, missing field, invalid URL)
- `500` — Internal server error (model failure, unexpected exception)

---

## API Reference

### `GET /`

Basic health check.

**Response `200`:**

```json
{
  "status": "running",
  "service": "AI Feature Extraction Service",
  "model": "MobileNetV2",
  "version": "2.0.0"
}
```

---

### `GET /health`

Detailed health check for monitoring and uptime checks.

**Response `200`:**

```json
{
  "status": "healthy",
  "service": "AI Feature Extraction Service",
  "model": "MobileNetV2 (Transfer Learning)",
  "endpoints": ["/extract", "/extract-url", "/health"],
  "feature_vector_size": 1280,
  "normalized_embeddings": true,
  "max_image_bytes": 8388608,
  "model_loaded": true,
  "version": "2.0.0"
}
```

---

### `POST /extract`

Extract a feature vector from an uploaded image file.

**Request:** `multipart/form-data`

| Field   | Type | Required | Description                        |
| ------- | ---- | -------- | ---------------------------------- |
| `image` | file | Yes      | Image file (JPEG, PNG, WebP, etc.) |

**Response `200` (success):**

```json
{
  "success": true,
  "features": [0.0234, -0.0112, 0.0891, ...],
  "feature_size": 1280,
  "normalized": true,
  "model": "MobileNetV2"
}
```

**Response `400` (invalid image):**

```json
{
  "success": false,
  "error": "Provided file is not a valid image"
}
```

**Response `400` (missing field):**

```json
{
  "success": false,
  "error": "No image provided"
}
```

---

### `POST /extract-url`

Extract a feature vector from an image at a remote URL.

**Request:** `application/json`

```json
{
  "url": "https://example.com/product-image.jpg"
}
```

**Response `200` (success):** Same format as `/extract`.

**Response `400` (invalid URL or non-image content):**

```json
{
  "success": false,
  "error": "Could not process image URL: URL did not return an image. Received content type: text/html"
}
```

---

## How It Integrates with the Backend

The Node.js backend communicates with this service through `src/utils/aiSearch.js`.

### Product Indexing

When a product is added or updated, the backend indexes it:

1. Sends the product image URL to `POST /extract-url`
2. Receives the 1280-dim feature vector
3. Stores the vector in the product's MongoDB document

### Visual Search

When a customer uploads a search image:

1. Backend receives the image upload
2. Sends it to `POST /extract`
3. Receives the query feature vector
4. Computes **cosine similarity** between the query vector and all indexed product vectors
5. Returns products ranked by similarity score

### Service URL

The backend expects the AI service at `http://localhost:5001` by default. This can be configured in the backend's environment variables.

---

## Common Development Tasks

### Testing an Endpoint Manually

**Health check:**

```bash
curl http://localhost:5001/health
```

**Extract from a local file:**

```bash
curl -X POST http://localhost:5001/extract \
  -F "image=@C:\path\to\ring.jpg"
```

**Extract from a URL:**

```bash
curl -X POST http://localhost:5001/extract-url \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"https://example.com/jewelry.jpg\"}"
```

### Checking if the Model is Loaded

```bash
curl http://localhost:5001/health | python -m json.tool
# Look for: "model_loaded": true
```

### Changing the Maximum Image Size

```bash
# Allow up to 16 MB
AI_SERVICE_MAX_IMAGE_BYTES=16777216 python app.py
```

### Changing the URL Fetch Timeout

```bash
# 30-second timeout for slow image URLs
AI_SERVICE_URL_TIMEOUT_SECONDS=30 python app.py
```

### Adding a New Endpoint

Open `app.py` and add a new route:

```python
@app.route("/my-endpoint", methods=["POST"])
def my_endpoint():
    """Description of what this endpoint does."""
    try:
        data = request.get_json(silent=True)
        # ... your logic ...
        return jsonify({"success": True, "result": "..."})
    except Exception as error:
        logger.error("Error in my_endpoint: %s", error)
        return build_error_payload(str(error), 500)
```

---

## Testing

Make sure your virtual environment is activated.

```bash
# Run all tests
python -m pytest tests/test_app.py -v

# Run with output (no capture)
python -m pytest tests/test_app.py -v -s
```

### Writing a New Test

Add test cases to `tests/test_app.py`:

```python
import pytest
from app import app, set_feature_model
import numpy as np

@pytest.fixture
def client():
    """Create a test client with a mock model."""
    class MockModel:
        def predict(self, x, verbose=0):
            # Return a non-zero vector so normalization works
            return np.ones((1, 1280), dtype=np.float32)

    set_feature_model(MockModel())
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_extract_no_image(client):
    response = client.post('/extract')
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`

Your virtual environment is not activated, or dependencies were not installed.

```bash
# Activate venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### `ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0`

TensorFlow 2.15 requires **Python 3.8–3.11**. Check your Python version:

```bash
python --version
```

If you have Python 3.12+, install Python 3.11 and create a new virtual environment with it:

```bash
py -3.11 -m venv venv    # Windows (if Python 3.11 is installed)
```

### Slow first request after startup

If `AI_SERVICE_PRELOAD_MODEL=0`, the model loads on the first request (~2–5 seconds). Set it to `1` (the default) to preload on startup and avoid this delay.

### `InvalidImageRequestError: Image exceeds maximum size`

The uploaded image is larger than the configured limit (default 8 MB). Options:

1. Compress the image before uploading
2. Increase the limit: `AI_SERVICE_MAX_IMAGE_BYTES=16777216 python app.py`

### `requests.exceptions.ConnectionError` when fetching image URLs

- The URL is unreachable from the server
- Check network connectivity
- Increase the timeout: `AI_SERVICE_URL_TIMEOUT_SECONDS=30`

### `FeatureExtractionError: Feature extractor returned a zero-length vector`

This is extremely rare and indicates the model returned an all-zero vector. Retry with a different image. If it happens consistently, the model may not have loaded correctly — restart the service.

### Port already in use

```bash
# Windows — find what's using port 5001
netstat -ano | findstr :5001

# Kill the process (replace PID)
taskkill /PID <PID> /F
```

Or change the port: `PORT=5002 python app.py`

### CORS errors from the browser

Flask-CORS is configured to allow all origins (`CORS(app)`). If you need to restrict it to specific origins, modify `app.py`:

```python
CORS(app, origins=["http://localhost:8081", "http://localhost:5000"])
```

### TensorFlow warnings about GPU / CUDA

These are informational warnings, not errors. TensorFlow will fall back to CPU automatically. You can suppress them:

```bash
# Windows CMD
set TF_CPP_MIN_LOG_LEVEL=2 & python app.py

# macOS / Linux
TF_CPP_MIN_LOG_LEVEL=2 python app.py
```

### `PIL.UnidentifiedImageError`

The uploaded file is not a valid image (could be a corrupted file, a PDF, or a non-image format). The service returns a `400` error with a descriptive message. Ensure the client sends a valid image file.
