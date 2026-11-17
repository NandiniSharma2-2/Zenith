"""
Zenith Sky Condition Monitoring — Streamlit Dashboard (V2)
A refined, astronomy-themed interface for night-sky cloud cover prediction,
measurable image indicator extraction, and observation suitability assessment.
"""

import os
import html
import csv
import hashlib
import io
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Ensure quiet logs & local cache directories for sandboxed execution
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Import core modular components
from predict import ZenithPredictor, OKTA_LABELS
from sky_analysis import analyze_sky_indicators, assess_astronomical_suitability

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ZENITH | Sky Observatory",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html_block(content: str) -> None:
    """Render HTML without Markdown treating indented fragments as code blocks."""
    normalized_content = textwrap.dedent(content)
    normalized_content = "\n".join(line.lstrip() for line in normalized_content.splitlines())
    st.markdown(normalized_content.strip(), unsafe_allow_html=True)


# --------------------------------------------------
# CUSTOM THEME — ZENITH OBSERVATORY
# --------------------------------------------------

html_block(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --gold: #e5bd78;
        --gold-bright: #f3d39b;
        --gold-muted: #a9824f;
        --cream: #eadcc3;
        --muted: #a8a8b2;
        --panel: rgba(8, 14, 25, .93);
        --panel-soft: rgba(13, 20, 33, .90);
        --line: rgba(201, 155, 89, .68);
    }

    html, body, [class*="css"] {
        font-family: 'Cormorant Garamond', Georgia, serif;
    }

    .stApp {
        color: var(--cream);
        background:
            linear-gradient(rgba(3, 7, 17, .38), rgba(3, 7, 17, .46)),
            url("zenith_background.png") center center / cover no-repeat fixed;
        background-color: #030711;
    }

    /* Restrained star field */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        opacity: .22;
        background-image:
            radial-gradient(1px 1px at 7% 13%, #fff 90%, transparent),
            radial-gradient(1px 1px at 18% 67%, #e8d7ad 90%, transparent),
            radial-gradient(2px 2px at 33% 21%, #fff 90%, transparent),
            radial-gradient(1px 1px at 48% 78%, #d9c6a0 90%, transparent),
            radial-gradient(1px 1px at 61% 12%, #fff 90%, transparent),
            radial-gradient(2px 2px at 74% 53%, #e8d7ad 90%, transparent),
            radial-gradient(1px 1px at 89% 27%, #fff 90%, transparent),
            radial-gradient(1px 1px at 96% 84%, #e8d7ad 90%, transparent);
        background-size: 420px 420px;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stDecoration"] { display: none; }

    .block-container {
        max-width: 1440px;
        padding: 1.45rem 2.4rem 1.1rem;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, p, label { color: var(--cream); }

    .masthead {
        text-align: center;
        padding: .1rem 0 1rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: .75rem;
    }
    .eyebrow {
        color: var(--gold);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: 2.4px;
        text-transform: uppercase;
        line-height: 1.3;
    }
    .eyebrow:before, .eyebrow:after {
        content: "";
        display: inline-block;
        width: min(12vw, 115px);
        height: 1px;
        vertical-align: middle;
        background: var(--gold-muted);
        margin: 0 25px 5px;
    }
    .main-title {
        color: var(--gold-bright);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: clamp(56px, 7vw, 90px);
        font-weight: 600;
        letter-spacing: .22em;
        line-height: .98;
        margin: .05rem 0 .12rem .22em;
        text-shadow: 0 2px 25px rgba(229, 189, 120, .15);
    }
    .subtitle {
        color: var(--gold-bright);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 19px;
        font-weight: 600;
        letter-spacing: .42em;
        margin-left: .42em;
        text-transform: uppercase;
    }
    .star-divider {
        color: var(--gold);
        font-size: 22px;
        letter-spacing: 8px;
        line-height: 1.2;
        margin: .25rem 0;
    }
    .intro {
        color: #e2c99f;
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 18px;
        font-style: italic;
        margin-top: .15rem;
    }

    .section-label {
        color: var(--gold);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: .1rem 0 .35rem;
    }
    .section-heading {
        color: var(--gold-bright);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 28px;
        font-weight: 700;
        line-height: 1.05;
        margin: 0 0 .05rem;
    }
    .section-description {
        color: #b9b6bd;
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 16px;
        margin-bottom: .5rem;
    }

    /* Upload surface */
    [data-testid="stFileUploader"] {
        margin-top: .15rem;
        margin-bottom: .85rem;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(9, 15, 26, .85);
        border: 1px dashed rgba(191, 151, 95, .65);
        border-radius: 12px;
        min-height: 72px;
        padding: .45rem .8rem;
    }
    [data-testid="stFileUploader"] label {
        color: #e6d4b5 !important;
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 17px;
    }
    [data-testid="stFileUploader"] small {
        color: #a9a8b1 !important;
    }
    [data-testid="stFileUploader"] button {
        color: var(--gold-bright) !important;
        border: 1px solid var(--gold-muted) !important;
        background: rgba(22, 29, 42, .92) !important;
    }

    /* Main dashboard panels */
    .panel {
        background: linear-gradient(145deg, rgba(11, 18, 30, .96), rgba(7, 13, 24, .95));
        border: 1px solid rgba(180, 135, 78, .58);
        border-radius: 13px;
        padding: 14px 15px 16px;
        min-height: 100%;
        box-shadow: 0 12px 36px rgba(0, 0, 0, .32), inset 0 1px rgba(255, 255, 255, .035);
        backdrop-filter: blur(12px);
    }
    .panel-heading {
        color: var(--gold);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 0 0 .45rem;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(201, 155, 89, .22);
    }
    .model-badge {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        background: rgba(229, 189, 120, .15);
        color: var(--gold-bright);
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(229, 189, 120, .35);
        text-transform: none;
        letter-spacing: 0.5px;
    }
    .image-frame {
        border: 1px solid #bd8e4e;
        border-radius: 10px;
        padding: 6px;
        background: rgba(0, 0, 0, .25);
        overflow: hidden;
    }
    [data-testid="stImage"] img {
        border-radius: 7px;
        object-fit: cover;
    }

    .analysis-top {
        display: grid;
        grid-template-columns: 1fr 1px 1fr;
        gap: 16px;
        align-items: center;
        min-height: 130px;
        padding: .25rem .5rem .35rem;
    }
    .analysis-divider {
        width: 1px;
        height: 90px;
        background: var(--gold-muted);
        justify-self: center;
    }
    .gauge-wrap { text-align: center; }
    .gauge {
        width: 142px;
        height: 82px;
        margin: 0 auto;
        position: relative;
        overflow: hidden;
    }
    .gauge svg { width: 142px; height: 142px; }
    .gauge-value {
        position: absolute;
        top: 34px;
        left: 0;
        width: 100%;
        color: var(--gold-bright);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 36px;
        line-height: 1;
    }
    .gauge-value span {
        font-size: 19px;
        color: var(--cream);
    }
    .gauge-caption {
        color: #d5c9b7;
        font-size: 13px;
        letter-spacing: .7px;
        margin-top: 1px;
        text-transform: uppercase;
    }
    .confidence-value {
        color: var(--gold-bright);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: clamp(38px, 4vw, 50px);
        font-weight: 600;
        line-height: 1;
        text-align: center;
    }
    .confidence-label {
        color: #e0d1b8;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .5px;
        text-align: center;
        margin: .15rem 0 .35rem;
        text-transform: uppercase;
    }
    .confidence-track {
        height: 8px;
        background: #343b48;
        border-radius: 20px;
        overflow: hidden;
        margin: 0 auto;
        width: 82%;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #c18a36, #f3d39b);
    }

    .assessment-title {
        color: var(--gold-bright);
        font-size: 16px;
        font-weight: 700;
        letter-spacing: .5px;
        margin: .55rem 0 .25rem;
        text-transform: uppercase;
    }
    .assessment-box {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        border: 1px solid rgba(181, 131, 62, .9);
        border-radius: 9px;
        background: linear-gradient(110deg, rgba(113, 82, 42, .26), rgba(20, 28, 42, .76));
        padding: 12px 14px;
        box-shadow: inset 0 1px rgba(255, 255, 255, .035);
    }
    .assessment-icon {
        font-size: 30px;
        line-height: 1;
        margin-top: 3px;
    }
    .assessment-status {
        color: #e9a93f;
        font-size: 16px;
        font-weight: 700;
        line-height: 1.1;
        text-transform: uppercase;
    }
    .assessment-copy {
        color: #eadcc3;
        font-size: 15px;
        line-height: 1.25;
        margin-top: 4px;
    }
    .recommendations-box {
        background: rgba(18, 25, 38, .75);
        border-left: 3px solid var(--gold);
        border-radius: 0 6px 6px 0;
        padding: 7px 10px;
        margin-top: 8px;
        font-size: 14px;
        color: #d8cebd;
    }

    .indicators-heading {
        color: var(--gold);
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: .75rem 0 .3rem;
        text-transform: uppercase;
    }
    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }
    .indicator-card {
        background: linear-gradient(145deg, rgba(26, 33, 47, .92), rgba(12, 18, 29, .94));
        border: 1px solid rgba(103, 111, 130, .2);
        border-radius: 8px;
        padding: 10px 6px;
        text-align: center;
        transition: border-color .2s ease, transform .2s ease;
    }
    .indicator-card:hover {
        border-color: rgba(229, 189, 120, .5);
        transform: translateY(-2px);
    }
    .indicator-name {
        color: #b9b7c1;
        font-size: 12px;
        letter-spacing: .5px;
        text-transform: uppercase;
    }
    .indicator-value {
        color: var(--gold-bright);
        font-size: 19px;
        line-height: 1.1;
        margin-top: 2px;
    }

    .probabilities-heading {
        color: var(--gold);
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: .75rem 0 .35rem;
        text-transform: uppercase;
    }
    .prob-bar-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 3px;
        font-size: 13px;
    }
    .prob-bar-label {
        width: 50px;
        color: #c4b69f;
        font-family: 'DM Mono', monospace;
    }
    .prob-bar-track {
        flex: 1;
        height: 6px;
        background: rgba(52, 59, 72, .6);
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #a9824f, #f3d39b);
        border-radius: 4px;
    }
    .prob-bar-fill-active {
        background: linear-gradient(90deg, #d49339, #ffd899);
    }
    .prob-bar-value {
        width: 45px;
        text-align: right;
        color: var(--gold-bright);
        font-family: 'DM Mono', monospace;
    }

    .disclaimer-note {
        color: #8c8f9b;
        font-family: 'DM Mono', monospace;
        font-size: 10.5px;
        line-height: 1.45;
        margin-top: .65rem;
        border-top: 1px dashed rgba(181, 135, 78, .35);
        padding-top: .45rem;
    }
    .awaiting {
        color: #c4b69f;
        border: 1px dashed rgba(185, 143, 83, .5);
        border-radius: 10px;
        background: rgba(9, 15, 26, .82);
        padding: 1.25rem;
        margin-top: .35rem;
        font-size: 16px;
        text-align: center;
    }
    .footer {
        border-top: 1px solid rgba(181, 135, 78, .65);
        color: #c29b61;
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1px;
        text-align: center;
        margin-top: 1.5rem;
        padding: .75rem 0 .25rem;
        text-transform: uppercase;
    }

    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem .75rem; }
        .main-title { font-size: clamp(42px, 13vw, 64px); }
        .eyebrow:before, .eyebrow:after { margin: 0 10px 5px; }
        .analysis-top { gap: 8px; padding-inline: 0; }
        .gauge, .gauge svg { width: 120px; }
        .indicator-grid { grid-template-columns: repeat(2, 1fr); }
        .panel { padding: 12px; }
        .model-badge { font-size: 9px; }
    }
    </style>
    """
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

html_block(
    """
    <div class="masthead">
        <div class="eyebrow">OBSERVATION SYSTEM // EST. 2026</div>
        <div class="main-title">ZENITH</div>
        <div class="subtitle">THE SKY OBSERVATORY</div>
        <div class="star-divider">──────────── ✦ ────────────</div>
        <div class="intro">Deep learning & automated sky condition monitoring for astronomy</div>
    </div>
    """
)

# --------------------------------------------------
# CACHED PREDICTOR INITIALIZATION
# --------------------------------------------------

@st.cache_resource(show_spinner="Initializing Zenith neural engine...")
def load_predictor() -> ZenithPredictor:
    """Load and warm up cached ZenithPredictor instance."""
    predictor = ZenithPredictor()
    predictor.warmup()
    return predictor


def render_gauge(value: int) -> str:
    """Render semicircular cloud-cover gauge for 0–8 oktas."""
    fraction = max(0.0, min(float(value) / 8.0, 1.0))
    circumference = 2 * np.pi * 55
    dash = circumference * fraction
    return textwrap.dedent(
        f"""
        <div class="gauge-wrap">
          <div class="gauge">
            <svg viewBox="0 0 142 142" aria-hidden="true">
              <circle cx="71" cy="71" r="55" fill="none"
                      stroke="#343b48" stroke-width="9"
                      stroke-dasharray="173 173"
                      transform="rotate(135 71 71)"
                      stroke-linecap="round"/>
              <circle cx="71" cy="71" r="55" fill="none"
                      stroke="url(#goldGradient)" stroke-width="9"
                      stroke-dasharray="{dash:.1f} {circumference:.1f}"
                      transform="rotate(135 71 71)"
                      stroke-linecap="round"/>
              <defs>
                <linearGradient id="goldGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="#c18a36"/>
                  <stop offset="100%" stop-color="#f3d39b"/>
                </linearGradient>
              </defs>
            </svg>
            <div class="gauge-value">{int(value)}<span> / 8</span></div>
          </div>
          <div class="gauge-caption">Oktas</div>
          <div class="gauge-caption" style="margin-top:10px;">Predicted Cloud Cover</div>
        </div>
        """
    ).strip()


def render_results(
    pred_res: Dict[str, Any],
    suitability: tuple,
    indicators: Dict[str, float],
) -> None:
    """Render formatted analysis cards, metrics, and probability bars."""
    cloud_cover = pred_res["cloud_cover"]
    expected_okta = pred_res["expected_okta"]
    confidence = pred_res["confidence"]
    probabilities = pred_res["probabilities"]
    model_name = pred_res["model_used"]

    status, explanation, details = suitability
    safe_status = html.escape(status)
    safe_explanation = html.escape(explanation)
    recommendations_html = ""
    if details.get("recommendations"):
        recs_str = "<br>• ".join([html.escape(r) for r in details["recommendations"]])
        recommendations_html = f'<div class="recommendations-box"><strong>🔭 Protocol:</strong><br>• {recs_str}</div>'

    confidence_width = max(0.0, min(float(confidence), 100.0))

    # Probability bars
    prob_rows = []
    for i, p in enumerate(probabilities):
        is_active = (i == cloud_cover)
        active_class = "prob-bar-fill-active" if is_active else "prob-bar-fill"
        font_weight = "bold" if is_active else "normal"
        row_str = textwrap.dedent(
            f"""
            <div class="prob-bar-row">
                <span class="prob-bar-label" style="font-weight:{font_weight}">Okta {i}</span>
                <div class="prob-bar-track">
                    <div class="{active_class}" style="width:{p*100:.1f}%"></div>
                </div>
                <span class="prob-bar-value" style="font-weight:{font_weight}">{p*100:.1f}%</span>
            </div>
            """
        ).strip()
        prob_rows.append(row_str)
    prob_bars_html = "".join(prob_rows)

    model_label = "V2 (Improved EfficientNetB0)" if "v2" in model_name else "V1 Baseline (MobileNetV2)"

    html_block(
        f"""
        <div class="panel">
            <div class="panel-heading">
                <span>03 // ANALYSIS & OBSERVATION RESULTS</span>
                <span class="model-badge">Model: {model_label}</span>
            </div>

            <div class="analysis-top">
                {render_gauge(cloud_cover)}
                <div class="analysis-divider"></div>
                <div>
                    <div class="confidence-value">{confidence:.1f}%</div>
                                <div class="confidence-label">Top-class score (not accuracy) &nbsp;✦&nbsp; Exp. {expected_okta:.1f} Oktas</div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width:{confidence_width:.2f}%"></div>
                    </div>
                </div>
            </div>

            <div class="assessment-title">✦ ASTRONOMICAL OBSERVATION SUITABILITY</div>
            <div class="assessment-box">
                <div class="assessment-icon">🔭</div>
                <div>
                    <div class="assessment-status">{safe_status}</div>
                    <div class="assessment-copy">{safe_explanation}</div>
                    {recommendations_html}
                </div>
            </div>

            <div class="indicators-heading">04 // MEASURABLE IMAGE INDICATORS</div>
            <div class="indicator-grid">
                <div class="indicator-card">
                    <div class="indicator-name">☼ Brightness</div>
                    <div class="indicator-value">{indicators['brightness']:.1f}</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">♢ Saturation</div>
                    <div class="indicator-value">{indicators['saturation']:.1f}</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">◐ Contrast</div>
                    <div class="indicator-value">{indicators['contrast']:.1f}</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-name">⚝ Sharpness</div>
                    <div class="indicator-value">{indicators.get('sharpness', 0.0):.1f}</div>
                </div>
            </div>

            <div class="probabilities-heading">05 // CLASS PROBABILITY DISTRIBUTION (0–8 OKTAS)</div>
            {prob_bars_html}

            <div class="disclaimer-note">
                <strong>Generalization note:</strong> The model scored 39.2% exact accuracy on the held-out C009 night set.
                This score is not a per-image guarantee. Other cameras, sensors, and ordinary sky photographs are unvalidated.
                The top-class score is a softmax output, not measured accuracy.
            </div>
            <div class="disclaimer-note">
                <strong>Scientific Note:</strong> Image indicators are computed directly from pixel sensor statistics
                (luminance, channel variance, and Laplacian gradients). They do not directly measure physical atmospheric seeing
                (arcseconds of optical turbulence) or calibrated sky background brightness (mag/arcsec²).
            </div>
        </div>
        """
    )


def load_observation_history() -> List[Dict[str, Any]]:
    """Load saved dashboard observations, failing visibly for malformed reports."""
    history_path = Path(__file__).resolve().parent / "reports" / "sky_observation_history.json"
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as history_file:
        history = json.load(history_file)
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError(f"Observation history must contain a JSON list of reports: {history_path}")
    return history


def save_observation_history(history: List[Dict[str, Any]]) -> None:
    """Persist the most recent observation reports for later trend analysis."""
    history_path = Path(__file__).resolve().parent / "reports" / "sky_observation_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history[-500:], history_file, indent=2)


def predict_obstruction_map(
    predictor: ZenithPredictor,
    image: Image.Image,
    grid_size: int = 5,
) -> List[List[Dict[str, Any]]]:
    """Estimate localized cloud obstruction by classifying image tiles as a batch."""
    width, height = image.size
    tiles = []
    for row in range(grid_size):
        for column in range(grid_size):
            left = round(column * width / grid_size)
            right = round((column + 1) * width / grid_size)
            top = round(row * height / grid_size)
            bottom = round((row + 1) * height / grid_size)
            tiles.append(image.crop((left, top, right, bottom)))

    predictions = predictor.predict_batch(tiles)
    obstruction_map = []
    for row in range(grid_size):
        map_row = []
        for column in range(grid_size):
            prediction = predictions[row * grid_size + column]
            oktas = prediction["cloud_cover"]
            obstruction = "CLEAR" if oktas <= 2 else "PARTIAL" if oktas <= 5 else "CLOUDY"
            map_row.append({
                "oktas": oktas,
                "obstruction": obstruction,
                "confidence": prediction["confidence"],
            })
        obstruction_map.append(map_row)
    return obstruction_map


def estimate_clear_window(history: List[Dict[str, Any]]) -> str:
    """Project a possible clear-sky start from recent observations, not a weather forecast."""
    if not history:
        return "Upload observations over time to estimate a potential clear-sky window."

    latest = history[-1]
    latest_oktas = int(latest["cloud_cover"])
    if latest_oktas <= 2:
        return "Clear conditions are observed now. How long they will last cannot be determined from these images."

    recent = history[-6:]
    if len(recent) < 3:
        return "At least three time-separated observations are needed before a trend can be estimated."

    times = [datetime.fromisoformat(item["timestamp"]).timestamp() for item in recent]
    elapsed_hours = [(timestamp - times[-1]) / 3600 for timestamp in times]
    if max(times) - min(times) < 15 * 60:
        return "Collect observations across at least 15 minutes to estimate a potential clear-sky window."

    oktas = [float(item["cloud_cover"]) for item in recent]
    slope = float(np.polyfit(elapsed_hours, oktas, 1)[0])
    if slope >= -0.05:
        return "Recent observations do not show clearing quickly enough to estimate a clear-sky start."

    hours_to_clear = (2.0 - latest_oktas) / slope
    if hours_to_clear <= 0 or hours_to_clear > 6:
        return "No clear-sky window is projected within the next six hours at the recent rate of change."

    clear_start = datetime.fromtimestamp(times[-1] + hours_to_clear * 3600).astimezone()
    return (
        f"If the recent clearing trend continues, the sky may reach 2 oktas in about "
        f"{clear_start.strftime('%H:%M')} local time. This is a short-term trend projection, not a weather forecast."
    )


def render_observatory_insights(
    predictor: ZenithPredictor,
    image: Image.Image,
    image_name: str,
    image_digest: str,
    prediction: Dict[str, Any],
    suitability: tuple,
    indicators: Dict[str, float],
) -> None:
    """Render the obstruction map, condition trends, outlook, alerts, and report history."""
    history_available = True
    try:
        history = load_observation_history()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read observation history: {error}")
        history = []
        history_available = False

    if not history or history[-1].get("image_digest") != image_digest:
        status, _, _ = suitability
        history.append({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image_name": Path(image_name).name,
            "image_digest": image_digest,
            "cloud_cover": int(prediction["cloud_cover"]),
            "cloud_condition": prediction["label"],
            "confidence": float(prediction["confidence"]),
            "suitability": status,
            "brightness": float(indicators["brightness"]),
        })
        history = history[-500:]
        if history_available:
            try:
                save_observation_history(history)
            except OSError as error:
                st.error(f"Could not save this observation report: {error}")

    current_oktas = int(prediction["cloud_cover"])
    previous_oktas = int(history[-2]["cloud_cover"]) if len(history) > 1 else None
    delta = current_oktas - previous_oktas if previous_oktas is not None else None

    st.markdown("### 06 // SKY WATCH")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Current sky condition",
        f"{current_oktas} / 8 oktas",
        delta=f"{delta:+d} oktas vs previous" if delta is not None else "First observation",
        delta_color="inverse",
    )
    metric_columns[1].metric("Top-class score (not accuracy)", f"{prediction['confidence']:.1f}%")
    metric_columns[2].metric("Observations saved", len(history))

    map_tab, changes_tab, outlook_tab, reports_tab = st.tabs(
        ["Cloud obstruction map", "Condition changes", "Clear-sky outlook", "Alerts & reports"]
    )

    with map_tab:
        map_cache_key = f"zenith_obstruction_map_{image_digest}"
        if map_cache_key not in st.session_state:
            with st.spinner("Mapping localized cloud obstruction..."):
                st.session_state[map_cache_key] = predict_obstruction_map(predictor, image)
        obstruction_map = st.session_state[map_cache_key]
        colors = {"CLEAR": "#3e9472", "PARTIAL": "#c18a36", "CLOUDY": "#b95757"}
        cells = []
        for row_index, row in enumerate(obstruction_map):
            for column_index, cell in enumerate(row):
                label = html.escape(
                    f"Sector {row_index + 1}, {column_index + 1}: "
                    f"{cell['obstruction'].lower()}, {cell['oktas']} oktas"
                )
                cells.append(
                    f'<div title="{label}" aria-label="{label}" style="'
                    f'background:{colors[cell["obstruction"]]};border-radius:5px;'
                    f'min-height:48px;display:grid;place-items:center;color:#fff;'
                    f'font-family:monospace;font-size:14px">{cell["oktas"]}</div>'
                )
        html_block(
            f"""
            <div style="display:grid;grid-template-columns:repeat(5,minmax(42px,1fr));
                gap:6px;max-width:430px;margin:12px 0">{''.join(cells)}</div>
            <div style="color:#d8cebd;font-size:14px">
                <span style="color:#62bd98">■ Clear (0–2)</span>&nbsp;&nbsp;
                <span style="color:#e5bd78">■ Partial (3–5)</span>&nbsp;&nbsp;
                <span style="color:#df7777">■ Cloudy (6–8)</span>
            </div>
            """
        )
        st.caption(
            "Each tile is classified independently by the sky model. Tile estimates are localized indicators; "
            "the full-image prediction above is the primary assessment."
        )

    with changes_tab:
        if history:
            chart_rows = history[-48:]
            chart_data = pd.DataFrame(
                {"Cloud cover (oktas)": [item["cloud_cover"] for item in chart_rows]},
                index=[item["timestamp"] for item in chart_rows],
            )
            st.line_chart(chart_data)
            if delta is not None:
                direction = "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"
                st.write(f"Cloud cover has {direction} by {abs(delta)} okta(s) since the previous saved image.")
            st.caption("Trend points are timestamped when images are analyzed; upload timing may differ from capture time.")
        else:
            st.info("No observation history is available yet.")

    with outlook_tab:
        st.write(estimate_clear_window(history))
        st.caption(
            "Clear-window estimates require repeated observations and extrapolate only the recent cloud-cover trend. "
            "They are not meteorological forecasts and cannot estimate how long a clear period will last."
        )
        clear_observations = [item for item in history if int(item["cloud_cover"]) <= 2][-10:]
        if clear_observations:
            st.write("Recent clear observations")
            st.dataframe(
                [
                    {"Analyzed at": item["timestamp"], "Image": item["image_name"], "Cloud cover": item["cloud_cover"]}
                    for item in clear_observations
                ],
                hide_index=True,
                use_container_width=True,
            )

    with reports_tab:
        alerts = []
        if current_oktas >= 6:
            alerts.append("Heavy cloud obstruction: optical observing conditions are poor.")
        elif current_oktas <= 2:
            alerts.append("Clear-sky conditions detected: consider starting the planned observation session.")
        if delta is not None and abs(delta) >= 2:
            direction = "increased" if delta > 0 else "decreased"
            alerts.append(f"Cloud cover {direction} by {abs(delta)} oktas since the previous report.")
        if prediction["confidence"] < 50:
            alerts.append("Low top-class score: verify the image and treat this assessment cautiously.")

        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("No observation alerts for the current image.")

        report_rows = [
            {
                "Analyzed at": item["timestamp"],
                "Image": item["image_name"],
                "Cloud cover (oktas)": item["cloud_cover"],
                "Top-class score (not accuracy, %)": item["confidence"],
                "Suitability": item["suitability"],
            }
            for item in reversed(history[-50:])
        ]
        st.dataframe(report_rows, hide_index=True, use_container_width=True)
        json_report = json.dumps(history, indent=2)
        csv_buffer = io.StringIO()
        if report_rows:
            writer = csv.DictWriter(csv_buffer, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download JSON history",
            data=json_report,
            file_name="zenith_observation_history.json",
            mime="application/json",
            use_container_width=True,
        )
        download_columns[1].download_button(
            "Download CSV report",
            data=csv_buffer.getvalue(),
            file_name="zenith_observation_history.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

html_block(
    """
    <div class="section-label">01 // IMAGE ACQUISITION</div>
    <div class="section-heading">Upload an all-sky or night-sky observation</div>
    <div class="section-description">Supports JPEG and PNG. Predictions are intended for all-sky camera images; ordinary sky photographs are outside the validated domain.</div>
    """
)

uploaded_file = st.file_uploader(
    "Drag and drop a sky observation image, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file is not None:
    try:
        # Load and validate image with safe stream reset
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.verify()  # Validate image integrity
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="medium")

        with left:
            html_block(
                """
                <div class="panel-heading">02 // ACQUIRED SKY OBSERVATION</div>
                """
            )
            st.image(image, caption="Acquired Observation Image", use_container_width=True)

        with right:
            with st.spinner("Analyzing sky condition & extracting optical indicators..."):
                predictor = load_predictor()
                pred_res = predictor.predict(image)
                indicators = analyze_sky_indicators(image)
                suitability = assess_astronomical_suitability(pred_res["cloud_cover"], indicators)
                render_results(pred_res, suitability, indicators)

        image_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        render_observatory_insights(
            predictor,
            image,
            uploaded_file.name,
            image_digest,
            pred_res,
            suitability,
            indicators,
        )

    except Exception as error:
        st.error(f"Analysis failed: {error}")
else:
    html_block(
        """
        <div class="awaiting">
            <div class="section-label">AWAITING OBSERVATION DATA</div>
            Upload an all-sky or night-sky observation to initiate the Zenith cloud-cover analysis system.
        </div>
        """
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

html_block(
    """
    <div class="footer">
        ZENITH OBSERVATORY SYSTEM &nbsp; ✦ &nbsp;
        SKY CONDITION MONITORING &nbsp; ✦ &nbsp; 2026
    </div>
    """
)
