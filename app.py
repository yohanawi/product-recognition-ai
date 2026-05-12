"""AI Feature Extraction Service for E-Commerce Platform."""

import io
import logging
import os
from typing import Optional

import numpy as np
import requests as http_requests
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "MobileNetV2"
MODEL_VERSION = "2.0.0"
FEATURE_VECTOR_SIZE = 1280
IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = int(os.getenv("AI_SERVICE_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
URL_TIMEOUT_SECONDS = int(os.getenv("AI_SERVICE_URL_TIMEOUT_SECONDS", "20"))
PORT = int(os.getenv("PORT", "5001"))

app = Flask(__name__)
CORS(app)

_feature_model = None


class InvalidImageRequestError(ValueError):
    """Raised when a request provides an invalid or unsupported image payload."""


class FeatureExtractionError(RuntimeError):
    """Raised when the model cannot produce a usable feature embedding."""


def should_preload_model() -> bool:
    return os.getenv("AI_SERVICE_PRELOAD_MODEL", "1").strip().lower() not in {"0", "false", "no"}


def get_resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def set_feature_model(model) -> None:
    global _feature_model
    _feature_model = model


def load_feature_model():
    logger.info("Loading %s model...", MODEL_NAME)
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )
    logger.info("%s model loaded successfully.", MODEL_NAME)
    return model


def get_feature_model():
    global _feature_model
    if _feature_model is None:
        _feature_model = load_feature_model()
    return _feature_model


def normalize_feature_vector(features: np.ndarray) -> np.ndarray:
    flattened = np.asarray(features, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(flattened)
    if norm <= 0:
        raise FeatureExtractionError("Feature extractor returned a zero-length vector")
    return flattened / norm


def prepare_image(image: Image.Image) -> np.ndarray:
    normalized = ImageOps.exif_transpose(image)

    if normalized.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", normalized.size, (255, 255, 255, 255))
        normalized = Image.alpha_composite(background, normalized.convert("RGBA")).convert("RGB")
    else:
        normalized = normalized.convert("RGB")

    fitted = ImageOps.fit(
        normalized,
        IMAGE_SIZE,
        method=get_resample_filter(),
        centering=(0.5, 0.5),
    )

    image_array = np.asarray(fitted, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def extract_features(image: Image.Image, model=None):
    try:
        input_batch = prepare_image(image)
        prediction_model = model or get_feature_model()
        features = prediction_model.predict(input_batch, verbose=0)
        normalized_features = normalize_feature_vector(features)
        return normalized_features.tolist()
    except Exception as error:
        logger.error("Feature extraction error: %s", error)
        raise


def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise InvalidImageRequestError("Empty image payload")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageRequestError(f"Image exceeds maximum size of {MAX_IMAGE_BYTES} bytes")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageRequestError("Provided file is not a valid image") from error


def read_uploaded_image(file_storage) -> Image.Image:
    filename = getattr(file_storage, "filename", "") or "upload"
    image_bytes = file_storage.read()
    image = read_image_from_bytes(image_bytes)
    logger.info("Processing uploaded image %s with size %s", filename, image.size)
    return image


def read_image_from_url(url: str) -> Image.Image:
    response = http_requests.get(url, timeout=URL_TIMEOUT_SECONDS, stream=True)
    response.raise_for_status()

    content_type = str(response.headers.get("content-type", "")).lower()
    if content_type and not content_type.startswith("image/"):
        raise InvalidImageRequestError(f"URL did not return an image. Received content type: {content_type}")

    content_length = response.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_IMAGE_BYTES:
        raise InvalidImageRequestError(f"Image exceeds maximum size of {MAX_IMAGE_BYTES} bytes")

    image = read_image_from_bytes(response.content)
    logger.info("Downloaded image from %s with size %s", url, image.size)
    return image


def build_feature_payload(features):
    return {
        "success": True,
        "features": features,
        "feature_size": len(features),
        "normalized": True,
        "model": MODEL_NAME,
    }


def build_error_payload(message: str, status_code: int):
    return jsonify({
        "success": False,
        "error": message,
    }), status_code

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "service": "AI Feature Extraction Service",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    })

@app.route("/extract", methods=["POST"])
def extract():
    """Extract normalized MobileNet features from an uploaded image."""
    try:
        if "image" not in request.files:
            return build_error_payload("No image provided", 400)

        file = request.files["image"]

        if file.filename == "":
            return build_error_payload("Empty filename", 400)

        image = read_uploaded_image(file)
        features = extract_features(image)

        logger.info("Features extracted successfully. Vector size: %s", len(features))
        return jsonify(build_feature_payload(features))
    except InvalidImageRequestError as error:
        logger.warning("Invalid extract request: %s", error)
        return build_error_payload(str(error), 400)
    except Exception as error:
        logger.error("Error processing extract request: %s", error)
        return build_error_payload(str(error), 500)

@app.route("/extract-url", methods=["POST"])
def extract_from_url():
    """Extract normalized MobileNet features from an image URL."""
    try:
        data = request.get_json(silent=True)
        if not data or "url" not in data:
            return build_error_payload("No URL provided in request body", 400)

        image_url = str(data["url"] or "").strip()
        if not image_url:
            return build_error_payload("URL must not be empty", 400)

        logger.info("Fetching image from URL: %s", image_url)
        image = read_image_from_url(image_url)
        features = extract_features(image)

        logger.info("Features extracted from URL. Vector size: %s", len(features))
        return jsonify(build_feature_payload(features))
    except (InvalidImageRequestError, http_requests.exceptions.RequestException) as error:
        logger.warning("Invalid extract-url request: %s", error)
        return build_error_payload(f"Could not process image URL: {error}", 400)
    except Exception as error:
        logger.error("Error processing URL request: %s", error)
        return build_error_payload(str(error), 500)

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check for monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "AI Feature Extraction Service",
        "model": f"{MODEL_NAME} (Transfer Learning)",
        "endpoints": ["/extract", "/extract-url", "/health"],
        "feature_vector_size": FEATURE_VECTOR_SIZE,
        "normalized_embeddings": True,
        "max_image_bytes": MAX_IMAGE_BYTES,
        "model_loaded": _feature_model is not None,
        "version": MODEL_VERSION,
    })

if should_preload_model():
    try:
        get_feature_model()
    except Exception as error:
        logger.exception("Failed to preload feature model: %s", error)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AI Feature Extraction Service")
    print("=" * 50)
    print(f"Running on: http://localhost:{PORT}")
    print(f"Model: {MODEL_NAME} (Transfer Learning)")
    print(f"Feature Vector Size: {FEATURE_VECTOR_SIZE} dimensions")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
