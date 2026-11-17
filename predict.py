"""
Zenith Sky Condition Monitoring — Prediction Pipeline Module
Provides cloud-cover inference (0–8 oktas) using the trained Zenith V2 model.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Sequence, Union, Optional
import numpy as np
from PIL import Image

# Ensure quiet TF logs & local cache directories for sandboxed and portable execution
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))

# Optional cv2 import (gracefully fallback to PIL + NumPy)
try:
    import cv2
except ImportError:
    cv2 = None

import tensorflow as tf

V2_MODEL_PATH = Path("models/zenith_v2.keras")
BASELINE_MODEL_PATH = Path("models/zenith_mobilenetv2.keras")

OKTA_LABELS = {
    0: "0 Oktas — Completely Clear Sky",
    1: "1 Okta — Nearly Clear (Isolated Clouds)",
    2: "2 Oktas — Scattered Clouds (Quarter Covered)",
    3: "3 Oktas — Scattered Clouds (Less than Half Covered)",
    4: "4 Oktas — Half Clouded Sky",
    5: "5 Oktas — Broken Clouds (More than Half Covered)",
    6: "6 Oktas — Broken Clouds (Three-Quarters Covered)",
    7: "7 Oktas — Nearly Overcast (Dense Cloud Cover)",
    8: "8 Oktas — Completely Overcast Sky",
}


class ZenithPredictor:
    """
    Inference manager for the Zenith Sky Condition Monitoring model.
    Loads the trained V2 model (or baseline fallback) and performs predictions.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is not None:
            chosen_path = Path(model_path)
            if not chosen_path.exists():
                raise FileNotFoundError(f"Specified model file does not exist: {chosen_path}")
            self.model_path = chosen_path
        elif V2_MODEL_PATH.exists():
            self.model_path = V2_MODEL_PATH
        elif BASELINE_MODEL_PATH.exists():
            self.model_path = BASELINE_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No trained Zenith model found. Checked:\n"
                f"- {V2_MODEL_PATH}\n"
                f"- {BASELINE_MODEL_PATH}\n"
                "Please run train.py to generate the V2 model checkpoint."
            )

        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        self.is_v2 = (self.model_path == V2_MODEL_PATH)

    def warmup(self):
        """Warm up graph execution once so subsequent inference calls are instant."""
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model(dummy, training=False)

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Load, validate, and resize input image to (224, 224, 3) in RGB."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Input image not found: {path_str}")
            with Image.open(path_str) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
                return np.array(img_resized, dtype=np.float32)
        elif isinstance(image_input, Image.Image):
            img_rgb = image_input.convert("RGB")
            img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            return np.array(img_resized, dtype=np.float32)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    img_pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
                elif image_input.shape[2] == 3:
                    img_pil = Image.fromarray(image_input.astype(np.uint8))
                else:
                    raise ValueError(f"Unsupported number of image channels: {image_input.shape[2]}")
            else:
                raise ValueError(f"Unsupported image array dimensions: {image_input.shape}")
            img_resized = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
            return np.array(img_resized, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Run inference on a single image.
        Returns:
        - cloud_cover: Discrete okta class (0–8)
        - expected_okta: Continuous expected okta value (weighted average of probabilities)
        - confidence: Maximum softmax output (%); not a calibrated accuracy estimate
        - probabilities: List of 9 float probabilities for classes 0 to 8
        - label: Human-readable okta description
        - model_used: Name of the model file employed
        """
        return self.predict_batch([image_input])[0]

    def predict_batch(
        self,
        image_inputs: Sequence[Union[str, Path, Image.Image, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Run one model call for several images, preserving the single-image result shape."""
        if not image_inputs:
            return []

        batch = np.stack([
            self.preprocess_image(image_input).astype(np.float32)
            for image_input in image_inputs
        ])
        raw_predictions = self.model(batch, training=False).numpy()
        results = []
        for raw_preds in raw_predictions:
            probs = [float(probability) for probability in raw_preds]
            predicted_class = int(np.argmax(raw_preds))
            confidence = float(raw_preds[predicted_class]) * 100.0
            expected_okta = float(np.sum([i * probability for i, probability in enumerate(probs)]))
            results.append({
                "cloud_cover": predicted_class,
                "expected_okta": round(expected_okta, 1),
                "confidence": round(confidence, 1),
                "probabilities": [round(probability, 4) for probability in probs],
                "label": OKTA_LABELS.get(predicted_class, f"{predicted_class} Oktas"),
                "model_used": self.model_path.name,
            })
        return results


# Singleton predictor instance for efficient caching
_PREDICTOR_INSTANCE: Optional[ZenithPredictor] = None


def get_predictor() -> ZenithPredictor:
    """Obtain or initialize the global predictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ZenithPredictor()
        _PREDICTOR_INSTANCE.warmup()
    return _PREDICTOR_INSTANCE


def predict_cloud_cover(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Convenience function for inference matching standard Zenith API."""
    predictor = get_predictor()
    return predictor.predict(image_input)


if __name__ == "__main__":
    import sys
    test_target = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    print(f"Testing prediction on: {test_target}")
    try:
        predictor = ZenithPredictor()
        predictor.warmup()
        res = predictor.predict(test_target)
        print(f"Model: {res['model_used']}")
        print(f"Predicted Cloud Cover: {res['cloud_cover']} / 8 oktas ({res['label']})")
        print(f"Expected Okta (Continuous): {res['expected_okta']} oktas")
        print(f"Top-class softmax score (not accuracy): {res['confidence']}%")
        print("Class Probabilities (0–8):")
        for i, p in enumerate(res['probabilities']):
            bar = "█" * int(p * 30)
            print(f"  Okta {i}: {p*100:5.1f}% | {bar}")
    except Exception as err:
        print(f"Prediction failed: {err}")