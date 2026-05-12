# AI Feature Extraction Service

## Overview

This microservice uses **Transfer Learning** with **MobileNetV2** CNN to extract normalized image embeddings for product similarity search.

For the full cross-project integration guide, backend endpoints, frontend behavior, deployment steps, and troubleshooting, see [backend/AI_PRODUCT_IDENTIFICATION.md](../backend/AI_PRODUCT_IDENTIFICATION.md).

## Architecture

- **Model**: MobileNetV2 (pretrained on ImageNet)
- **Output**: 1280-dimensional L2-normalized feature vector
- **Framework**: TensorFlow + Keras
- **API**: Flask REST API
- **Preprocessing**: EXIF-aware orientation, alpha-channel flattening, aspect-preserving crop to 224x224

## Installation

### 1. Install Python Dependencies

```bash
cd ai-service
pip install -r requirements.txt
```

Optional environment variables:

```bash
set AI_SERVICE_PRELOAD_MODEL=1
set AI_SERVICE_MAX_IMAGE_BYTES=8388608
set AI_SERVICE_URL_TIMEOUT_SECONDS=20
set PORT=5001
```

### 2. Run the Service

```bash
python app.py
```

Service will run on: `http://localhost:5001`

## Tests

```bash
python -m unittest tests.test_app
```

## API Endpoints

### `GET /`

Health check endpoint

**Response:**

```json
{
  "status": "running",
  "service": "AI Feature Extraction Service",
  "model": "MobileNetV2",
  "version": "2.0.0"
}
```

### `POST /extract`

Extract features from an image

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: image file with key "image"

**Response:**

```json
{
  "success": true,
  "features": [0.123, 0.456, ...],
  "feature_size": 1280,
  "normalized": true,
  "model": "MobileNetV2"
}
```

Validation behavior:

- rejects files larger than `AI_SERVICE_MAX_IMAGE_BYTES` with `400`
- rejects invalid image payloads with `400`
- returns `500` if the feature extractor cannot produce a usable embedding

### `POST /extract-url`

Extract features from a remote image URL.

**Request:**

```json
{
  "url": "https://example.com/image.jpg"
}
```

The service validates content type, enforces the configured maximum image size, and uses the same preprocessing path as uploaded files.

### `GET /health`

Detailed readiness and configuration status.

**Response:**

```json
{
  "status": "healthy",
  "service": "AI Feature Extraction Service",
  "model": "MobileNetV2 (Transfer Learning)",
  "feature_vector_size": 1280,
  "normalized_embeddings": true,
  "max_image_bytes": 8388608,
  "model_loaded": true,
  "version": "2.0.0"
}
```

## How It Works

1. **Image Upload** → User uploads image via API
2. **Preprocessing** → Image is orientation-corrected, alpha-safe, center-cropped to 224x224, and normalized
3. **Feature Extraction** → MobileNetV2 CNN processes image
4. **Feature Vector** → Returns 1280-dimensional normalized embedding
5. **Storage** → Features saved to MongoDB with product
6. **Similarity Search** → Cosine similarity finds similar products

## Technical Details

- **Transfer Learning**: Uses pretrained weights from ImageNet (1M+ images)
- **No Training Required**: Model already trained, saves time and resources
- **Fast**: MobileNet optimized for speed (~10ms per image)
- **Academically Strong**: Industry-standard approach for image search
- **Stable Similarity**: Embeddings are normalized before returning, which keeps cosine matching consistent across backend and admin indexing flows

## Environment Variables

- `PORT` default: `5001`
- `AI_SERVICE_MAX_IMAGE_BYTES` default: `8388608`
- `AI_SERVICE_URL_TIMEOUT_SECONDS` default: `20`
- `AI_SERVICE_PRELOAD_MODEL` default: `1`
 
