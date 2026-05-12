import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

os.environ.setdefault("AI_SERVICE_PRELOAD_MODEL", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module


class DummyModel:
    def predict(self, batch, verbose=0):
        return np.ones((1, 1280), dtype=np.float32)


class ZeroVectorModel:
    def predict(self, batch, verbose=0):
        return np.zeros((1, 1280), dtype=np.float32)


def build_image_bytes(mode="RGB", size=(360, 240), color=(180, 120, 90)):
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class AIServiceAppTests(unittest.TestCase):
    def setUp(self):
        app_module.set_feature_model(DummyModel())
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.set_feature_model(None)

    def test_extract_returns_normalized_feature_vector(self):
        response = self.client.post(
            "/extract",
            data={"image": (build_image_bytes(), "sample.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        self.assertTrue(payload["success"])
        self.assertEqual(payload["feature_size"], 1280)
        self.assertTrue(payload["normalized"])

        vector = np.array(payload["features"], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_extract_rejects_missing_image(self):
        response = self.client.post("/extract", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertIn("No image provided", response.get_json()["error"])

    def test_extract_rejects_invalid_image_payload(self):
        response = self.client.post(
            "/extract",
            data={"image": (io.BytesIO(b"not-an-image"), "broken.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid image", response.get_json()["error"])

    def test_extract_rejects_oversized_payload(self):
        original_limit = app_module.MAX_IMAGE_BYTES
        app_module.MAX_IMAGE_BYTES = 32

        try:
            response = self.client.post(
                "/extract",
                data={"image": (io.BytesIO(b"x" * 64), "too-large.bin")},
                content_type="multipart/form-data",
            )
        finally:
            app_module.MAX_IMAGE_BYTES = original_limit

        self.assertEqual(response.status_code, 400)
        self.assertIn("maximum size", response.get_json()["error"])

    def test_extract_url_rejects_non_image_content(self):
        fake_response = Mock()
        fake_response.headers = {"content-type": "text/html"}
        fake_response.content = b"<html></html>"
        fake_response.raise_for_status = Mock()

        with patch.object(app_module.http_requests, "get", return_value=fake_response):
            response = self.client.post("/extract-url", json={"url": "https://example.com/not-image"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not process image URL", response.get_json()["error"])

    def test_extract_url_requires_request_body_url(self):
        response = self.client.post("/extract-url", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("No URL provided", response.get_json()["error"])

    def test_extract_url_rejects_oversized_content_length(self):
        original_limit = app_module.MAX_IMAGE_BYTES
        app_module.MAX_IMAGE_BYTES = 128

        fake_response = Mock()
        fake_response.headers = {"content-type": "image/png", "content-length": "1024"}
        fake_response.content = build_image_bytes().getvalue()
        fake_response.raise_for_status = Mock()

        try:
            with patch.object(app_module.http_requests, "get", return_value=fake_response):
                response = self.client.post("/extract-url", json={"url": "https://example.com/huge-image.png"})
        finally:
            app_module.MAX_IMAGE_BYTES = original_limit

        self.assertEqual(response.status_code, 400)
        self.assertIn("maximum size", response.get_json()["error"])

    def test_extract_url_maps_request_timeouts_to_client_errors(self):
        with patch.object(
            app_module.http_requests,
            "get",
            side_effect=app_module.http_requests.exceptions.Timeout("timed out"),
        ):
            response = self.client.post("/extract-url", json={"url": "https://example.com/slow-image.png"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not process image URL", response.get_json()["error"])

    def test_extract_url_returns_normalized_feature_vector(self):
        fake_response = Mock()
        fake_response.headers = {"content-type": "image/png", "content-length": "1024"}
        fake_response.content = build_image_bytes().getvalue()
        fake_response.raise_for_status = Mock()

        with patch.object(app_module.http_requests, "get", return_value=fake_response):
            response = self.client.post("/extract-url", json={"url": "https://example.com/image.png"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["feature_size"], 1280)
        self.assertTrue(payload["normalized"])

    def test_extract_returns_500_for_zero_vector(self):
        app_module.set_feature_model(ZeroVectorModel())

        response = self.client.post(
            "/extract",
            data={"image": (build_image_bytes(), "sample.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("zero-length vector", response.get_json()["error"])

    def test_health_reports_model_and_feature_size(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["feature_vector_size"], 1280)
        self.assertIn("model", payload)

    def test_prepare_image_handles_alpha_channel(self):
        prepared = app_module.prepare_image(Image.open(build_image_bytes(mode="RGBA", color=(100, 40, 200, 120))))

        self.assertEqual(prepared.shape, (1, 224, 224, 3))


if __name__ == "__main__":
    unittest.main()