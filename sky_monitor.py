#!/usr/bin/env python3
"""
Sky Condition Monitor - Simplified version
Analyzes sky images and provides sky condition reports for astronomical observation.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

class SkyConditionMonitor:
    """
    Simple sky condition monitoring system that analyzes sky images
    and provides observation suitability assessments.
    """
    
    OKTA_DESCRIPTIONS = {
        0: "Completely Clear Sky",
        1: "Nearly Clear (Isolated Clouds)",
        2: "Scattered Clouds (Quarter Covered)",
        3: "Scattered Clouds (Less than Half Covered)", 
        4: "Half Clouded Sky",
        5: "Broken Clouds (More than Half Covered)",
        6: "Broken Clouds (Three-Quarters Covered)",
        7: "Nearly Overcast (Dense Cloud Cover)",
        8: "Completely Overcast Sky"
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.history_file = self.data_dir / "sky_history.json"
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to numpy array."""
        with Image.open(image_path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    
    def analyze_basic_indicators(self, image_array: np.ndarray) -> Dict[str, float]:
        """Extract basic image indicators without complex dependencies."""
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Image must be RGB array")
        
        # Resize for consistent analysis
        pil_img = Image.fromarray(image_array).resize((512, 512), Image.Resampling.BILINEAR)
        resized = np.array(pil_img, dtype=np.uint8)
        
        # Convert to grayscale
        gray = 0.299 * resized[:, :, 0].astype(np.float32) + \
               0.587 * resized[:, :, 1].astype(np.float32) + \
               0.114 * resized[:, :, 2].astype(np.float32)
        
        # Calculate indicators
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Simple cloudiness heuristic based on pixel intensity distribution
        hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
        uniformity = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0
        
        # Edge detection via simple gradient
        grad_x = np.diff(gray, axis=1)[:-1, :]
        grad_y = np.diff(gray, axis=0)[:, :-1]
        edge_strength = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        return {
            "brightness": float(round(brightness, 2)),
            "contrast": float(round(contrast, 2)),
            "uniformity_index": float(round(uniformity, 3)),
            "edge_strength": float(round(edge_strength, 2))
        }
    
    def estimate_cloud_coverage(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate cloud coverage based on image indicators.
        Returns oktas (0-8) and confidence.
        """
        brightness = indicators.get("brightness", 0)
        contrast = indicators.get("contrast", 0)
        uniformity = indicators.get("uniformity_index", 0)
        
        # Simple heuristic estimation (simplified from Zenith's approach)
        if brightness < 30:
            # Very dark - likely clear night
            if contrast > 10:
                okta = 0  # Clear with stars visible
            else:
                okta = 1  # Clear but uniform
        elif brightness < 100:
            # Moderate brightness
            if uniformity < 0.5:
                okta = 2  # Scattered clouds
            else:
                okta = 4  # Partly cloudy
        else:
            # Bright - likely cloudy or daytime
            if contrast < 5:
                okta = 8  # Overcast
            elif contrast < 15:
                okta = 6  # Broken clouds
            else:
                okta = 4  # Partly cloudy
        
        # Adjust based on contrast (higher contrast often means clearer sky)
        if contrast > 20:
            okta = max(0, okta - 1)
        
        # Confidence based on indicator consistency
        confidence = 0.7  # Base confidence
        
        return {
            "oktas": float(okta),
            "confidence": float(confidence),
            "description": self.OKTA_DESCRIPTIONS.get(int(okta), "Unknown")
        }
    
    def assess_observation_suitability(self, cloud_coverage: Dict[str, float], 
                                      indicators: Dict[str, float]) -> Dict[str, any]:
        """Assess suitability for astronomical observation."""
        oktas = cloud_coverage.get("oktas", 8)
        brightness = indicators.get("brightness", 255)
        
        if oktas <= 2:
            suitability = "EXCELLENT"
            reasoning = "Clear or scattered cloud conditions ideal for most observations"
        elif oktas <= 4:
            suitability = "GOOD"
            reasoning = "Partially cloudy but usable for bright targets and some deep-sky work"
        elif oktas <= 6:
            suitability = "POOR"
            reasoning = "Heavy cloud coverage limits observation to bright targets only"
        else:
            suitability = "UNSUITABLE"
            reasoning = "Overcast conditions prevent meaningful astronomical observation"
        
        # Adjust for brightness (daytime vs nighttime)
        if brightness > 100:
            suitability = "DAYTIME"
            reasoning = "Daytime conditions - only solar/lunar observation possible"
        
        return {
            "suitability": suitability,
            "reasoning": reasoning,
            "recommendations": self._generate_recommendations(suitability, oktas)
        }
    
    def _generate_recommendations(self, suitability: str, oktas: float) -> List[str]:
        """Generate observation recommendations."""
        recs = []
        
        if suitability == "EXCELLENT":
            recs.extend([
                "Ideal for deep-sky photography",
                "Perfect for planetary observation",
                "Excellent conditions for variable star work",
                "Consider long-exposure astrophotography"
            ])
        elif suitability == "GOOD":
            recs.extend([
                "Good for lunar and planetary observation",
                "Suitable for brighter deep-sky objects",
                "Consider short-exposure astrophotography",
                "Monitor for changing conditions"
            ])
        elif suitability == "POOR":
            recs.extend([
                "Focus on bright targets (Moon, planets)",
                "Avoid deep-sky observation",
                "Use for equipment testing only",
                "Wait for clearing conditions"
            ])
        elif suitability == "UNSUITABLE":
            recs.extend([
                "No astronomical observation recommended",
                "Use time for equipment maintenance",
                "Plan for future observation sessions",
                "Check weather forecasts for clearing"
            ])
        elif suitability == "DAYTIME":
            recs.extend([
                "Solar observation with proper filters only",
                "Lunar observation if visible",
                "Daytime astronomy outreach activities",
                "Equipment calibration and testing"
            ])
        
        return recs
    
    def generate_cloud_coverage_map(self, image_array: np.ndarray, grid_size: int = 8) -> Dict[str, any]:
        """Generate simple obstruction map by dividing image into grid."""
        height, width = image_array.shape[:2]
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        obstruction_map = []
        
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                y_start = i * grid_h
                x_start = j * grid_w
                patch = image_array[y_start:y_start+grid_h, x_start:x_start+grid_w]
                
                # Simple cloudiness estimate per patch
                patch_gray = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
                patch_contrast = np.std(patch_gray)
                
                if patch_contrast < 5:
                    obstruction = "CLEAR"
                elif patch_contrast < 15:
                    obstruction = "PARTIAL"
                else:
                    obstruction = "CLOUDY"
                
                row.append({
                    "x": j,
                    "y": i,
                    "obstruction": obstruction,
                    "contrast": float(round(patch_contrast, 2))
                })
            obstruction_map.append(row)
        
        return {
            "grid_size": grid_size,
            "cell_height": grid_h,
            "cell_width": grid_w,
            "map": obstruction_map
        }
    
    def load_historical_data(self) -> List[Dict]:
        """Load historical sky condition data."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def save_to_history(self, analysis: Dict):
        """Save current analysis to history."""
        history = self.load_historical_data()
        
        analysis_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **analysis
        }
        
        history.append(analysis_with_timestamp)
        
        # Keep only last 100 entries
        if len(history) > 100:
            history = history[-100:]
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def analyze_sky_image(self, image_path: str) -> Dict[str, any]:
        """Complete analysis of a sky image."""
        try:
            # Load image
            image_array = self.load_image(image_path)
            
            # Analyze indicators
            indicators = self.analyze_basic_indicators(image_array)
            
            # Estimate cloud coverage
            cloud_coverage = self.estimate_cloud_coverage(indicators)
            
            # Generate obstruction map
            obstruction_map = self.generate_cloud_coverage_map(image_array)
            
            # Assess observation suitability
            suitability = self.assess_observation_suitability(cloud_coverage, indicators)
            
            # Check for clear-sky intervals from history
            historical_data = self.load_historical_data()
            clear_intervals = self._analyze_clear_intervals(historical_data)
            
            # Prepare full report
            report = {
                "timestamp": datetime.now().isoformat(),
                "image_file": image_path,
                "image_dimensions": list(image_array.shape[:2]),
                "indicators": indicators,
                "cloud_coverage": cloud_coverage,
                "obstruction_map": obstruction_map,
                "observation_suitability": suitability,
                "clear_sky_intervals": clear_intervals,
                "alerts": self._generate_alerts(cloud_coverage, indicators, historical_data)
            }
            
            # Save to history
            self.save_to_history(report)
            
            return report
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_clear_intervals(self, historical_data: List[Dict]) -> List[Dict]:
        """Analyze historical data for clear sky intervals."""
        clear_intervals = []
        
        for i, entry in enumerate(historical_data[-10:]):  # Last 10 entries
            coverage = entry.get("cloud_coverage", {})
            oktas = coverage.get("oktas", 8)
            
            if oktas <= 2:
                interval = {
                    "timestamp": entry.get("timestamp", "unknown"),
                    "duration_minutes": 30,  # Estimated
                    "oktas": oktas,
                    "suitability": "CLEAR"
                }
                clear_intervals.append(interval)
        
        return clear_intervals
    
    def _generate_alerts(self, cloud_coverage: Dict[str, float], 
                        indicators: Dict[str, float],
                        historical_data: List[Dict]) -> List[str]:
        """Generate alerts based on current conditions and trends."""
        alerts = []
        
        oktas = cloud_coverage.get("oktas", 8)
        brightness = indicators.get("brightness", 0)
        
        # Sudden change alert
        if historical_data:
            last_entry = historical_data[-1]
            last_coverage = last_entry.get("cloud_coverage", {})
            last_oktas = last_coverage.get("oktas", 8)
            
            if abs(oktas - last_oktas) >= 3:
                alerts.append(f"Rapid cloud coverage change: {last_oktas} → {oktas} oktas")
        
        # Clear sky alert
        if oktas <= 1:
            alerts.append("Excellent observing conditions - clear skies detected")
        
        # Deterioration alert
        if oktas >= 6:
            alerts.append("Poor observing conditions - heavy cloud coverage")
        
        # Daytime alert
        if brightness > 150:
            alerts.append("Daytime conditions - limited observation options")
        
        return alerts
    
    def print_report(self, report: Dict):
        """Print formatted sky condition report."""
        print("\n" + "="*60)
        print("SKY CONDITION MONITORING REPORT")
        print("="*60)
        
        print(f"\n📅 Timestamp: {report.get('timestamp', 'N/A')}")
        print(f"📁 Image: {report.get('image_file', 'N/A')}")
        print(f"📐 Dimensions: {report.get('image_dimensions', 'N/A')}")
        
        print("\n🌤️ CLOUD COVERAGE ANALYSIS")
        print("-"*40)
        coverage = report.get('cloud_coverage', {})
        print(f"Cloud Coverage: {coverage.get('oktas', 'N/A')} oktas")
        print(f"Description: {coverage.get('description', 'N/A')}")
        print(f"Confidence: {coverage.get('confidence', 'N/A')*100:.1f}%")
        
        print("\n📊 IMAGE INDICATORS")
        print("-"*40)
        indicators = report.get('indicators', {})
        for key, value in indicators.items():
            print(f"{key.title()}: {value}")
        
        print("\n🔭 OBSERVATION SUITABILITY")
        print("-"*40)
        suitability = report.get('observation_suitability', {})
        print(f"Status: {suitability.get('suitability', 'N/A')}")
        print(f"Reasoning: {suitability.get('reasoning', 'N/A')}")
        print("\nRecommendations:")
        for rec in suitability.get('recommendations', []):
            print(f"  • {rec}")
        
        print("\n🗺️ CLOUD OBSTRUCTION MAP")
        print("-"*40)
        obs_map = report.get('obstruction_map', {})
        grid = obs_map.get('map', [])
        for row in grid:
            row_str = ""
            for cell in row:
                status = cell.get('obstruction', 'UNKNOWN')
                if status == "CLEAR":
                    row_str += "○ "
                elif status == "PARTIAL":
                    row_str += "◑ "
                else:
                    row_str += "● "
            print(f"  {row_str}")
        print("  ○ Clear  ◑ Partial  ● Cloudy")
        
        print("\n⏱️ CLEAR SKY INTERVALS")
        print("-"*40)
        intervals = report.get('clear_sky_intervals', [])
        if intervals:
            for interval in intervals:
                print(f"  • {interval.get('timestamp', 'N/A')}: {interval.get('suitability', 'N/A')} "
                      f"({interval.get('oktas', 'N/A')} oktas)")
        else:
            print("  No recent clear sky intervals detected")
        
        print("\n⚠️ ALERTS & WARNINGS")
        print("-"*40)
        alerts = report.get('alerts', [])
        if alerts:
            for alert in alerts:
                print(f"  ⚠ {alert}")
        else:
            print("  No alerts at this time")
        
        print("\n" + "="*60)


def main():
    """Main function to analyze a sky image."""
    monitor = SkyConditionMonitor()
    
    # Check for test image
    test_image = Path("test.jpg")
    if not test_image.exists():
        # Try to find any image in the data directory
        data_dir = Path("data")
        if data_dir.exists():
            image_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.jpeg")) + list(data_dir.rglob("*.png"))
            if image_files:
                image_path = str(image_files[0])
                print(f"Using image: {image_path}")
            else:
                print("No test image found. Please provide an image path as argument.")
                print("Usage: python sky_monitor.py <image_path>")
                return
        else:
            print("No test image found. Please provide an image path as argument.")
            print("Usage: python sky_monitor.py <image_path>")
            return
    else:
        image_path = str(test_image)
    
    # Analyze image
    print(f"Analyzing sky image: {image_path}")
    report = monitor.analyze_sky_image(image_path)
    
    if "error" in report:
        print(f"Error analyzing image: {report['error']}")
        return
    
    # Print report
    monitor.print_report(report)
    
    # Save full report to file
    output_file = f"sky_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()