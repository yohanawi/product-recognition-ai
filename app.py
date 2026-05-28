"""AI image service for jewelry classification and visual search."""

import io
import json
import logging
import os
from pathlib import Path
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
CLASSIFIER_NAME = "JewelryClassifier"
MODEL_VERSION = "3.0.0"
FEATURE_VECTOR_SIZE = 1280
IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = int(os.getenv("AI_SERVICE_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
URL_TIMEOUT_SECONDS = int(os.getenv("AI_SERVICE_URL_TIMEOUT_SECONDS", "20"))
PORT = int(os.getenv("PORT", "5001"))
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("AI_SERVICE_MODEL_DIR", BASE_DIR / "models"))
CLASSIFIER_PATH = Path(os.getenv("AI_CLASSIFIER_PATH", MODEL_DIR / "jewelry_classifier.keras"))
LEGACY_CLASSIFIER_PATH = MODEL_DIR / "jewelry_classifier.h5"
CLASS_NAMES_PATH = Path(os.getenv("AI_CLASS_NAMES_PATH", MODEL_DIR / "class_names.json"))

app = Flask(__name__)
CORS(app)

_feature_model = None
_classifier_model = None
_class_names = None


class InvalidImageRequestError(ValueError):
    """Raised when a request provides an invalid or unsupported image payload."""


class FeatureExtractionError(RuntimeError):
    """Raised when the model cannot produce a usable feature embedding."""


class ClassifierUnavailableError(RuntimeError):
    """Raised when prediction is requested before the classifier is available."""


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
    logger.info("Loading %s feature model...", MODEL_NAME)
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )
    logger.info("%s feature model loaded successfully.", MODEL_NAME)
    return model


def get_feature_model():
    global _feature_model
    if _feature_model is None:
        _feature_model = load_feature_model()
    return _feature_model


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        logger.warning("Class names file not found: %s", CLASS_NAMES_PATH)
        return []

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise ValueError("class_names.json must contain a JSON array of strings")

    return names


def validate_uploaded_file(file_storage) -> None:
    if file_storage is None:
        raise InvalidImageRequestError("No image provided")

    filename = getattr(file_storage, "filename", "") or ""
    if not filename.strip():
        raise InvalidImageRequestError("Empty filename")


def get_class_names():
    global _class_names
    if _class_names is None:
        _class_names = load_class_names()
    return _class_names


def resolve_classifier_path() -> Optional[Path]:
    if CLASSIFIER_PATH.exists():
        return CLASSIFIER_PATH
    if LEGACY_CLASSIFIER_PATH.exists():
        logger.warning("Using legacy .h5 classifier. Retrain to create %s.", CLASSIFIER_PATH)
        return LEGACY_CLASSIFIER_PATH
    return None


def load_classifier_model():
    classifier_path = resolve_classifier_path()
    if classifier_path is None:
        raise ClassifierUnavailableError(
            f"Classifier model not found. Expected {CLASSIFIER_PATH}."
        )

    logger.info("Loading classifier model from %s...", classifier_path)
    model = tf.keras.models.load_model(classifier_path, compile=False)
    logger.info("Classifier loaded successfully.")
    return model


def get_classifier_model():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = load_classifier_model()
    return _classifier_model


def normalize_feature_vector(features: np.ndarray) -> np.ndarray:
    flattened = np.asarray(features, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(flattened)
    if norm <= 0:
        raise FeatureExtractionError("Feature extractor returned a zero-length vector")
    return flattened / norm


def normalize_class_label(label: str) -> str:
    normalized = str(label or "").strip().replace("_", " ").replace("-", " ").lower()
    known_singular = {
        "rings": "ring",
        "bracelets": "bracelet",
        "necklaces": "necklace",
        "earrings": "earring",
        "pendants": "pendant",
        "bangles": "bangle",
    }
    return known_singular.get(normalized, normalized)


def prepare_rgb_image(image: Image.Image) -> Image.Image:
    normalized = ImageOps.exif_transpose(image)

    if normalized.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", normalized.size, (255, 255, 255, 255))
        normalized = Image.alpha_composite(background, normalized.convert("RGBA")).convert("RGB")
    else:
        normalized = normalized.convert("RGB")

    return ImageOps.fit(
        normalized,
        IMAGE_SIZE,
        method=get_resample_filter(),
        centering=(0.5, 0.5),
    )


def prepare_feature_image(image: Image.Image) -> np.ndarray:
    fitted = prepare_rgb_image(image)
    image_array = np.asarray(fitted, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Backward-compatible alias for the feature extractor input tensor."""
    return prepare_feature_image(image)


def prepare_classifier_image(image: Image.Image) -> np.ndarray:
    fitted = prepare_rgb_image(image)
    image_array = np.asarray(fitted, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def extract_features(image: Image.Image, model=None):
    try:
        input_batch = prepare_feature_image(image)
        prediction_model = model or get_feature_model()
        features = prediction_model.predict(input_batch, verbose=0)
        normalized_features = normalize_feature_vector(features)
        return normalized_features.tolist()
    except Exception as error:
        logger.error("Feature extraction error: %s", error)
        raise


def predict_category(image: Image.Image):
    model = get_classifier_model()
    class_names = get_class_names()
    if not class_names:
        raise ClassifierUnavailableError("class_names.json is missing or empty")

    probabilities = model.predict(prepare_classifier_image(image), verbose=0)[0]
    probabilities = np.asarray(probabilities, dtype=np.float32)
    ranked_indices = np.argsort(probabilities)[::-1]

    top_categories = []
    for index in ranked_indices[: min(5, len(ranked_indices))]:
        raw_label = class_names[int(index)]
        normalized_name = normalize_class_label(raw_label)
        top_categories.append({
            "name": normalized_name,
            "category": normalized_name,
            "label": raw_label,
            "confidence": round(float(probabilities[int(index)]), 6),
        })

    best = top_categories[0]
    return {
        "category": best["name"],
        "label": best["label"],
        "confidence": best["confidence"],
        "top_categories": top_categories,
    }


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
    validate_uploaded_file(file_storage)
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


def build_prediction_payload(prediction, features):
    return {
        "success": True,
        "prediction": prediction,
        "features": {
            "vector": features,
            "feature_size": len(features),
            "normalized": True,
            "model": MODEL_NAME,
        },
        "classifier": {
            "model": CLASSIFIER_NAME,
            "path": str(resolve_classifier_path() or CLASSIFIER_PATH),
            "class_count": len(get_class_names()),
        },
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
        "service": "AI Jewelry Image Service",
        "feature_model": MODEL_NAME,
        "classifier": CLASSIFIER_NAME,
        "version": MODEL_VERSION,
    })


@app.route("/extract", methods=["POST"])
def extract():
    """Extract normalized MobileNet features from an uploaded image."""
    try:
        file = request.files.get("image")

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


@app.route("/predict", methods=["POST"])
def predict():
    """Predict jewelry category and return the normalized visual-search vector."""
    try:
        file = request.files.get("image")

        image = read_uploaded_image(file)
        prediction = predict_category(image)
        features = extract_features(image)

        logger.info(
            "Prediction completed for category=%s confidence=%.4f feature_size=%s",
            prediction["category"],
            prediction["confidence"],
            len(features),
        )

        return jsonify(build_prediction_payload(prediction, features))
    except InvalidImageRequestError as error:
        logger.warning("Invalid predict request: %s", error)
        return build_error_payload(str(error), 400)
    except ClassifierUnavailableError as error:
        logger.error("Classifier unavailable: %s", error)
        return build_error_payload(str(error), 503)
    except Exception as error:
        logger.error("Error processing predict request: %s", error)
        return build_error_payload(str(error), 500)


@app.route("/health", methods=["GET"])
def health():
    """Detailed health check for monitoring."""
    classifier_path = resolve_classifier_path()
    class_names = get_class_names()
    return jsonify({
        "status": "healthy",
        "service": "AI Jewelry Image Service",
        "model": f"{MODEL_NAME} (Transfer Learning Embeddings)",
        "classifier": {
            "available": classifier_path is not None and bool(class_names),
            "path": str(classifier_path) if classifier_path else str(CLASSIFIER_PATH),
            "class_count": len(class_names),
            "classes": [normalize_class_label(name) for name in class_names],
            "loaded": _classifier_model is not None,
        },
        "endpoints": ["/extract", "/extract-url", "/predict", "/health"],
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

    try:
        if resolve_classifier_path() is not None:
            get_classifier_model()
    except Exception as error:
        logger.exception("Failed to preload classifier model: %s", error)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AI Jewelry Image Service")
    print("=" * 50)
    print(f"Running on: http://localhost:{PORT}")
    print(f"Feature Model: {MODEL_NAME}")
    print(f"Classifier Model: {resolve_classifier_path() or CLASSIFIER_PATH}")
    print(f"Feature Vector Size: {FEATURE_VECTOR_SIZE} dimensions")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
