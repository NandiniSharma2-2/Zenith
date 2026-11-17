#!/usr/bin/env python3
"""
Demonstration of Zenith V2 Sky Condition Monitoring Project
Shows the capabilities and outputs without requiring full dependencies.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

def demonstrate_project():
    """Demonstrate the key capabilities of the Zenith V2 project."""
    
    print("="*70)
    print("ZENITH V2 - SKY CONDITION MONITORING PROJECT DEMONSTRATION")
    print("="*70)
    
    # 1. Show project structure
    print("\n📁 PROJECT STRUCTURE")
    print("-"*40)
    project_files = [
        "app.py - Streamlit observatory dashboard",
        "predict.py - Cloud cover prediction pipeline", 
        "sky_analysis.py - Image indicator extraction",
        "train.py - Two-stage model training",
        "evaluate.py - Benchmark evaluation",
        "sky_monitor.py - Simplified analysis tool (created)",
        "requirements.txt - Dependencies"
    ]
    for file in project_files:
        print(f"  • {file}")
    
    # 2. Show data directory structure
    print("\n📊 DATA STRUCTURE")
    print("-"*40)
    data_dir = Path("data/GOA-ALLSKY/ALLSKY-GOA")
    if data_dir.exists():
        subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"  Data directory contains: {len(subdirs)} subdirectories")
        print(f"  Subdirectories: {', '.join(subdirs[:5])}{'...' if len(subdirs) > 5 else ''}")
    else:
        print("  Data directory not found")
    
    # 3. Show model performance from evaluation reports
    print("\n📈 MODEL PERFORMANCE (from evaluation reports)")
    print("-"*40)
    
    eval_file = Path("reports/evaluation_results.json")
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        v2_night = eval_data.get("v2_night", {})
        baseline_night = eval_data.get("baseline_night", {})
        
        print(f"  Baseline Model Nighttime Accuracy: {baseline_night.get('accuracy', 0)*100:.1f}%")
        print(f"  Zenith V2 Nighttime Accuracy: {v2_night.get('accuracy', 0)*100:.1f}%")
        print(f"  Improvement: +{(v2_night.get('accuracy', 0) - baseline_night.get('accuracy', 0))*100:.1f}%")
        print(f"  V2 Okta MAE: {v2_night.get('okta_mae', 0):.2f} oktas")
        print(f"  V2 Within ±1 Okta Accuracy: {v2_night.get('adjacent_accuracy', 0)*100:.1f}%")
    else:
        print("  Evaluation data not found")
    
    # 4. Show sample sky analysis
    print("\n🔭 SAMPLE SKY ANALYSIS")
    print("-"*40)
    
    test_image = Path("test.jpg")
    if test_image.exists():
        try:
            # Simple analysis without OpenCV
            img = Image.open(test_image)
            img_array = np.array(img.convert("RGB"), dtype=np.uint8)
            
            # Calculate basic indicators
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            print(f"  Test Image: {test_image.name}")
            print(f"  Dimensions: {img_array.shape[1]}x{img_array.shape[0]} pixels")
            print(f"  Brightness: {brightness:.1f}")
            print(f"  Contrast: {contrast:.1f}")
            
            # Simple cloudiness estimate
            if brightness < 50:
                condition = "Very dark - likely clear night"
            elif brightness < 100:
                condition = "Moderate - scattered clouds"
            else:
                condition = "Bright - likely cloudy or daytime"
            
            print(f"  Condition Estimate: {condition}")
            
        except Exception as e:
            print(f"  Error analyzing image: {e}")
    else:
        print("  Test image not found")
    
    # 5. Show sky reports
    print("\n📋 AVAILABLE REPORTS")
    print("-"*40)
    
    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*"))
        for report in report_files[:5]:
            print(f"  • {report.name} ({report.stat().st_size:,} bytes)")
        if len(report_files) > 5:
            print(f"  ... and {len(report_files) - 5} more files")
    else:
        print("  Reports directory not found")
    
    # 6. Show the sky monitor report we generated
    print("\n🔄 RECENT SKY MONITOR ANALYSIS")
    print("-"*40)
    
    sky_reports = list(Path(".").glob("sky_report_*.json"))
    if sky_reports:
        latest_report = max(sky_reports, key=lambda x: x.stat().st_mtime)
        with open(latest_report, 'r') as f:
            report_data = json.load(f)
        
        coverage = report_data.get("cloud_coverage", {})
        suitability = report_data.get("observation_suitability", {})
        
        print(f"  Latest Report: {latest_report.name}")
        print(f"  Cloud Coverage: {coverage.get('oktas', 'N/A')} oktas")
        print(f"  Description: {coverage.get('description', 'N/A')}")
        print(f"  Observation Status: {suitability.get('suitability', 'N/A')}")
        print(f"  Reasoning: {suitability.get('reasoning', 'N/A')}")
    else:
        print("  No sky monitor reports found")
    
    # 7. Key features summary
    print("\n🌟 KEY PROJECT FEATURES")
    print("-"*40)
    features = [
        "Machine learning model for cloud cover classification (0-8 oktas)",
        "Domain-invariant training for day/night and cross-camera generalization",
        "Image indicator extraction: brightness, contrast, saturation, sharpness",
        "Astronomical observation suitability assessment",
        "Cloud obstruction mapping across sky regions",
        "Streamlit dashboard for interactive analysis",
        "Continuous expected okta metric for fine-grained assessment",
        "Rigorous benchmarking on held-out test data"
    ]
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
    
    # 8. How to use the project
    print("\n🚀 HOW TO USE THE PROJECT")
    print("-"*40)
    usage_steps = [
        "1. Install dependencies: pip install -r requirements.txt",
        "2. Run Streamlit dashboard: streamlit run app.py",
        "3. Upload sky images for analysis",
        "4. View cloud coverage predictions and indicators",
        "5. Check observation suitability assessment",
        "6. Use sky_monitor.py for command-line analysis"
    ]
    for step in usage_steps:
        print(f"  {step}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    demonstrate_project()