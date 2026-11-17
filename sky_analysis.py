"""
Zenith Sky Condition Monitoring — Image Analysis & Astronomical Suitability Module
Extracts image quality indicators and evaluates astronomical observation suitability.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Union, Any
import numpy as np
from PIL import Image

# Ensure quiet TF/Matplotlib logs & local cache directories
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))

# Optional cv2 import (gracefully fallback to PIL + NumPy)
try:
    import cv2
except ImportError:
    cv2 = None


def _to_rgb_array(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Normalize input into an RGB float32/uint8 numpy array."""
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image not found at path: {path_str}")
        with Image.open(path_str) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return np.stack([image_input.astype(np.uint8)] * 3, axis=-1)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                return image_input[:, :, :3].astype(np.uint8)
            elif image_input.shape[2] == 3:
                return image_input.astype(np.uint8)
            else:
                raise ValueError(f"Invalid image array channels: {image_input.shape[2]}")
        else:
            raise ValueError(f"Invalid image array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def analyze_sky_indicators(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, float]:
    """
    Extract quantifiable image indicators from an all-sky or night-sky image:
    - Brightness: Mean Value channel in HSV color space [0, 255]
    - Saturation: Mean Saturation channel in HSV color space [0, 255]
    - Contrast: Standard deviation of pixel intensities in grayscale
    - Sharpness: Variance of the Laplacian filter (indicator of blur/focus/stars)
    - Haze Index: Heuristic estimate based on Dark Channel Prior proxy [0.0, 1.0]
    """
    rgb = _to_rgb_array(image_input)

    # Resize to standard analysis resolution (512, 512)
    if cv2 is not None:
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        pil_img = Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR)
        resized = np.array(pil_img, dtype=np.uint8)

    # Brightness, Saturation, Contrast
    if cv2 is not None:
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float64)
        brightness = float(np.mean(hsv[:, :, 2]))
        saturation = float(np.mean(hsv[:, :, 1]))
        contrast = float(np.std(gray))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))
    else:
        # Pure NumPy / PIL fallback
        r = resized[:, :, 0].astype(np.float64)
        g = resized[:, :, 1].astype(np.float64)
        b = resized[:, :, 2].astype(np.float64)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        contrast = float(np.std(gray))

        max_ch = np.maximum(np.maximum(r, g), b)
        min_ch = np.minimum(np.minimum(r, g), b)
        brightness = float(np.mean(max_ch))

        with np.errstate(divide="ignore", invalid="ignore"):
            sat = np.where(max_ch > 0, ((max_ch - min_ch) / max_ch) * 255.0, 0.0)
        saturation = float(np.mean(sat))

        # Discrete 2D Laplacian operator in pure NumPy
        lap = (
            gray[2:, 1:-1] + gray[:-2, 1:-1] +
            gray[1:-1, 2:] + gray[1:-1, :-2] -
            4.0 * gray[1:-1, 1:-1]
        )
        sharpness = float(np.var(lap))

    # Empirical haze proxy: ratio of minimum channel to maximum channel
    min_c = np.min(resized, axis=2).astype(np.float32)
    max_c = np.maximum(np.max(resized, axis=2).astype(np.float32), 1.0)
    haze_ratio = float(np.mean(min_c / max_c))

    return {
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "haze_index": round(haze_ratio, 3),
    }


# Backwards compatibility alias
def analyze_sky(image_path: Union[str, Path, Image.Image]) -> Dict[str, float]:
    return analyze_sky_indicators(image_path)


def assess_astronomical_suitability(
    cloud_cover_oktas: int,
    indicators: Dict[str, float],
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Evaluate astronomical observing suitability based on predicted cloud cover
    and measurable image indicators using clear, transparent domain rules.

    Returns:
    - status: 'SUITABLE', 'PARTIALLY SUITABLE', or 'NOT SUITABLE'
    - explanation: Human-readable rationale for the assessment
    - details: Structured metadata including contributing factors and recommendations
    """
    brightness = indicators.get("brightness", 0.0)
    contrast = indicators.get("contrast", 0.0)
    sharpness = indicators.get("sharpness", 0.0)
    haze = indicators.get("haze_index", 0.0)

    factors = []
    recommendations = []

    # 1. Cloud Cover evaluation (Primary criterion)
    if cloud_cover_oktas <= 2:
        cloud_status = "clear_to_scattered"
        factors.append(f"Low cloud fraction ({cloud_cover_oktas}/8 oktas)")
    elif cloud_cover_oktas <= 5:
        cloud_status = "partially_cloudy"
        factors.append(f"Moderate cloud fraction ({cloud_cover_oktas}/8 oktas)")
    else:
        cloud_status = "overcast"
        factors.append(f"Heavy cloud obstruction ({cloud_cover_oktas}/8 oktas)")

    # 2. Ambient brightness evaluation
    high_illumination = brightness > 135.0
    moderate_illumination = brightness > 90.0
    if high_illumination:
        factors.append(f"High background luminance ({brightness:.1f})")
    elif moderate_illumination:
        factors.append(f"Moderate background luminance ({brightness:.1f})")

    # 3. Contrast & Sharpness evaluation
    low_contrast = contrast < 25.0
    if low_contrast:
        factors.append(f"Low image contrast ({contrast:.1f}), suggesting uniform veil or haze")

    is_soft = sharpness < 35.0
    if is_soft and cloud_cover_oktas <= 3:
        factors.append("Low gradient sharpness (possible atmospheric diffusion or defocus)")

    # Final decision logic
    if cloud_status == "clear_to_scattered":
        if high_illumination:
            status = "PARTIALLY SUITABLE"
            explanation = (
                f"Sky is largely clear ({cloud_cover_oktas}/8 oktas), but high background luminance "
                f"({brightness:.1f}) indicates strong ambient illumination (e.g. Moon, twilight, or light pollution). "
                "Faint deep-sky objects will suffer from reduced contrast; lunar and planetary observations remain viable."
            )
            recommendations.append("Prioritize bright planetary, lunar, or double-star targets over deep-sky imaging.")
        elif low_contrast:
            status = "PARTIALLY SUITABLE"
            explanation = (
                f"Low cloud cover ({cloud_cover_oktas}/8 oktas) detected, but low scene contrast ({contrast:.1f}) "
                "indicates pervasive atmospheric haze or thin cirrus veils that degrade contrast."
            )
            recommendations.append("Use light-pollution or narrow-band filters; check seeing stability before long exposures.")
        else:
            status = "SUITABLE FOR OBSERVATION"
            explanation = (
                f"Optimal observation window: minimal cloud coverage ({cloud_cover_oktas}/8 oktas) with "
                f"balanced background luminance ({brightness:.1f}) and sharp contrast ({contrast:.1f})."
            )
            recommendations.append("Excellent window for both deep-sky imaging and wide-field astrophotography.")

    elif cloud_status == "partially_cloudy":
        status = "PARTIALLY SUITABLE"
        explanation = (
            f"Partially obstructed sky ({cloud_cover_oktas}/8 oktas). Transient cloud bands are likely to "
            "interfere with continuous tracking or calibrated photometry. Short exposures and patchy gaps may be usable."
        )
        recommendations.append("Target open sky sectors away from moving cloud banks; monitor drift intervals.")

    else:  # overcast
        status = "NOT SUITABLE"
        explanation = (
            f"Severe cloud obstruction ({cloud_cover_oktas}/8 oktas). The optical observing path is heavily or "
            "completely blocked, preventing effective astronomical data acquisition."
        )
        recommendations.append("Suspend optical observations; consider closing observatory dome or cap to protect optics.")

    details = {
        "status": status,
        "cloud_oktas": cloud_cover_oktas,
        "contributing_factors": factors,
        "recommendations": recommendations,
        "indicators": indicators,
        "disclaimer": (
            "Zenith indicators are camera-derived optical heuristics (luminance, contrast, and Laplacian gradient). "
            "They do not directly quantify physical atmospheric seeing (arcsecond wavefront turbulence) or calibrated "
            "photometric sky brightness (mag/arcsec²), which require dedicated DIMM or SQM hardware."
        ),
    }

    return status, explanation, details
