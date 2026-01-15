"""
AeroRisk - Problem 5: Advanced Recommendation Engine
=====================================================
GOAL: Generate actionable safety recommendations based on risk factors

Features:
- Rule-based recommendations with SHAP integration
- SMS (Safety Management System) alignment
- Cost-benefit analysis
- Priority ranking
- Expected risk reduction calculations
- Regulatory compliance mapping

Author: Umang Kumar
Date: 2024-01-14
"""

# ============================================
# STEP 1: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import shap
import joblib
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("💡 AeroRisk - Advanced Recommendation Engine")
print("=" * 60)


# ============================================
# STEP 2: Define Recommendation Types
# ============================================

class Priority(Enum):
    """Recommendation priority levels."""
    CRITICAL = "CRITICAL"  # Immediate action required
    HIGH = "HIGH"          # Action within 24 hours
    MEDIUM = "MEDIUM"      # Action within 7 days
    LOW = "LOW"            # Discretionary improvement


class SMSPillar(Enum):
    """ICAO SMS (Safety Management System) Pillars."""
    POLICY = "Safety Policy & Objectives"
    RISK_MGMT = "Safety Risk Management"
    ASSURANCE = "Safety Assurance"
    PROMOTION = "Safety Promotion"


class CostLevel(Enum):
    """Implementation cost levels."""
    MINIMAL = "$"          # < $1,000
    LOW = "$$"             # $1,000 - $10,000
    MEDIUM = "$$$"         # $10,000 - $100,000
    HIGH = "$$$$"          # $100,000 - $1,000,000
    VERY_HIGH = "$$$$$"    # > $1,000,000


@dataclass
class Recommendation:
    """A safety recommendation with full context."""
    id: str
    title: str
    description: str
    priority: str
    risk_reduction_percent: float
    cost_level: str
    implementation_time: str
    sms_pillar: str
    regulatory_refs: List[str]
    shap_contribution: float  # How much this factor contributed
    trigger_value: float      # The value that triggered this
    trigger_threshold: float  # The threshold that was exceeded
    action_steps: List[str]
    kpis: List[str]          # Key Performance Indicators to track


# ============================================
# STEP 3: Recommendation Rules Database
# ============================================

RECOMMENDATION_RULES = {
    # ===================== WEATHER RULES =====================
    "WEATHER_SEVERE": {
        "feature": "weather_risk_score",
        "condition": lambda x: x > 70,
        "title": "Delay Flight - Severe Weather Conditions",
        "description": "Weather conditions exceed safe operating limits. Consider delaying operations until conditions improve.",
        "priority": Priority.CRITICAL,
        "risk_reduction": 35.0,
        "cost": CostLevel.LOW,
        "time": "Immediate",
        "sms_pillar": SMSPillar.RISK_MGMT,
        "regulations": ["14 CFR 91.103", "ICAO Annex 3"],
        "actions": [
            "Monitor weather updates every 30 minutes",
            "Brief crew on alternate airports",
            "Calculate additional fuel for diversions",
            "Notify passengers of potential delay"
        ],
        "kpis": ["On-time departure rate", "Weather-related incidents"]
    },
    
    "WEATHER_MODERATE": {
        "feature": "weather_risk_score",
        "condition": lambda x: 50 < x <= 70,
        "title": "Enhanced Weather Monitoring Required",
        "description": "Elevated weather risk detected. Implement enhanced monitoring procedures.",
        "priority": Priority.HIGH,
        "risk_reduction": 15.0,
        "cost": CostLevel.MINIMAL,
        "time": "1-2 hours",
        "sms_pillar": SMSPillar.ASSURANCE,
        "regulations": ["14 CFR 91.103"],
        "actions": [
            "Review latest METAR/TAF",
            "Confirm alternate airport fuel loaded",
            "Brief crew on weather conditions",
            "Check PIREPs for route"
        ],
        "kpis": ["Weather briefing completion rate"]
    },
    
    # ===================== VISIBILITY RULES =====================
    "LOW_VISIBILITY": {
        "feature": "visibility_m",
        "condition": lambda x: x < 3000,
        "title": "Low Visibility Operations Protocol",
        "description": "Visibility below standard minimums. Implement LVO procedures if equipped.",
        "priority": Priority.CRITICAL,
        "risk_reduction": 40.0,
        "cost": CostLevel.MEDIUM,
        "time": "Immediate",
        "sms_pillar": SMSPillar.RISK_MGMT,
        "regulations": ["14 CFR 91.175", "ICAO Annex 6"],
        "actions": [
            "Verify crew is LVO certified",
            "Confirm aircraft CAT II/III equipped",
            "Check RVR requirements",
            "Coordinate with ATC for low vis ops"
        ],
        "kpis": ["Low visibility go-around rate", "LVO training currency"]
    },
    
    # ===================== WIND RULES =====================
    "HIGH_WIND": {
        "feature": "wind_speed_kt",
        "condition": lambda x: x > 25,
        "title": "High Wind Crosswind Landing Assessment",
        "description": "Wind speeds exceed normal limits. Evaluate crosswind component and aircraft limitations.",
        "priority": Priority.HIGH,
        "risk_reduction": 25.0,
        "cost": CostLevel.MINIMAL,
        "time": "30 minutes",
        "sms_pillar": SMSPillar.RISK_MGMT,
        "regulations": ["Aircraft POH/AFM"],
        "actions": [
            "Calculate crosswind component",
            "Review aircraft crosswind limits",
            "Consider alternate runway",
            "Plan for go-around"
        ],
        "kpis": ["Hard landing rate", "Go-around rate"]
    },
    
    # ===================== CREW FATIGUE RULES =====================
    "CREW_FATIGUE_HIGH": {
        "feature": "crew_fatigue_index",
        "condition": lambda x: x > 80,
        "title": "CRITICAL: Crew Fatigue Limit Exceeded",
        "description": "Crew fatigue levels are dangerously high. Immediate crew rest or replacement required.",
        "priority": Priority.CRITICAL,
        "risk_reduction": 45.0,
        "cost": CostLevel.MEDIUM,
        "time": "Immediate",
        "sms_pillar": SMSPillar.POLICY,
        "regulations": ["14 CFR 117", "EASA ORO.FTL"],
        "actions": [
            "Verify actual rest hours",
            "Consider crew replacement",
            "Delay departure for minimum rest",
            "Document fatigue report"
        ],
        "kpis": ["Fatigue-related incidents", "FDP utilization rate"]
    },
    
    "CREW_FATIGUE_MODERATE": {
        "feature": "crew_fatigue_index",
        "condition": lambda x: 60 < x <= 80,
        "title": "Crew Fatigue Management Actions",
        "description": "Elevated crew fatigue detected. Implement fatigue countermeasures.",
        "priority": Priority.HIGH,
        "risk_reduction": 20.0,
        "cost": CostLevel.MINIMAL,
        "time": "Before flight",
        "sms_pillar": SMSPillar.RISK_MGMT,
        "regulations": ["14 CFR 117"],
        "actions": [
            "Brief crew on fatigue countermeasures",
            "Schedule controlled rest if applicable",
            "Reduce workload where possible",
            "Monitor crew performance"
        ],
        "kpis": ["Fatigue report submission rate"]
    },
    
    # ===================== DUTY HOURS RULES =====================
    "EXCESSIVE_DUTY": {
        "feature": "crew_duty_hours",
        "condition": lambda x: x > 12,
        "title": "Flight Duty Period Limit Approaching",
        "description": "Crew approaching or exceeding FDP limits. Plan for extension or relief.",
        "priority": Priority.HIGH,
        "risk_reduction": 30.0,
        "cost": CostLevel.LOW,
        "time": "1-2 hours",
        "sms_pillar": SMSPillar.POLICY,
        "regulations": ["14 CFR 117.17", "14 CFR 117.19"],
        "actions": [
            "Calculate remaining FDP",
            "Assess need for extension",
            "Pre-position relief crew if needed",
            "Document FDP extension justification"
        ],
        "kpis": ["FDP extension rate", "Max FDP utilization"]
    },
    
    # ===================== MAINTENANCE RULES =====================
    "MAINTENANCE_OVERDUE": {
        "feature": "maintenance_overdue_flag",
        "condition": lambda x: x == 1,
        "title": "CRITICAL: Maintenance Action Overdue",
        "description": "Aircraft has overdue maintenance items. Ground aircraft until resolved.",
        "priority": Priority.CRITICAL,
        "risk_reduction": 50.0,
        "cost": CostLevel.HIGH,
        "time": "Immediate",
        "sms_pillar": SMSPillar.ASSURANCE,
        "regulations": ["14 CFR 43.3", "14 CFR 91.409"],
        "actions": [
            "Ground aircraft immediately",
            "Review overdue items with maintenance",
            "Complete required maintenance",
            "Document MEL/CDL items"
        ],
        "kpis": ["Maintenance completion rate", "AOG hours"]
    },
    
    "MAINTENANCE_RISK_HIGH": {
        "feature": "maintenance_risk",
        "condition": lambda x: x > 70,
        "title": "High Maintenance Risk - Enhanced Inspection",
        "description": "Elevated maintenance risk factors. Conduct enhanced pre-flight inspection.",
        "priority": Priority.HIGH,
        "risk_reduction": 25.0,
        "cost": CostLevel.LOW,
        "time": "1-2 hours",
        "sms_pillar": SMSPillar.ASSURANCE,
        "regulations": ["14 CFR 91.403"],
        "actions": [
            "Complete extended walk-around",
            "Review recent maintenance history",
            "Verify open MEL items",
            "Check fluid levels and tire pressure"
        ],
        "kpis": ["Pre-flight finding rate", "Maintenance write-up rate"]
    },
    
    # ===================== AIRCRAFT AGE RULES =====================
    "OLD_AIRCRAFT": {
        "feature": "aircraft_age_years",
        "condition": lambda x: x > 25,
        "title": "Aging Aircraft Enhanced Monitoring",
        "description": "Aircraft age exceeds 25 years. Implement aging aircraft monitoring procedures.",
        "priority": Priority.MEDIUM,
        "risk_reduction": 15.0,
        "cost": CostLevel.MEDIUM,
        "time": "Ongoing",
        "sms_pillar": SMSPillar.ASSURANCE,
        "regulations": ["14 CFR 121.1107", "AC 120-104"],
        "actions": [
            "Review CPCP compliance",
            "Check structural modifications history",
            "Verify fatigue-critical inspections current",
            "Document corrosion prevention status"
        ],
        "kpis": ["Structural finding rate", "Corrosion finding rate"]
    },
    
    # ===================== SCHEDULE RULES =====================
    "RUSHED_TURNAROUND": {
        "feature": "turnaround_time_mins",
        "condition": lambda x: x < 30,
        "title": "Rushed Turnaround - Risk Mitigation",
        "description": "Turnaround time below minimum. Risk of incomplete checks.",
        "priority": Priority.HIGH,
        "risk_reduction": 20.0,
        "cost": CostLevel.MINIMAL,
        "time": "Immediate",
        "sms_pillar": SMSPillar.RISK_MGMT,
        "regulations": ["Company SOP"],
        "actions": [
            "Prioritize critical safety checks",
            "Use quick-turn checklist",
            "Verify fuel, weight, balance complete",
            "Accept minor delay if needed for safety"
        ],
        "kpis": ["Turnaround incident rate", "Ground damage rate"]
    },
    
    # ===================== INJURY/SEVERITY RULES =====================
    "HIGH_INJURY_PATTERN": {
        "feature": "injury_score",
        "condition": lambda x: x > 50,
        "title": "High-Risk Pattern - Enhanced Safety Measures",
        "description": "Historical injury patterns indicate elevated risk. Implement enhanced safety measures.",
        "priority": Priority.HIGH,
        "risk_reduction": 30.0,
        "cost": CostLevel.MEDIUM,
        "time": "Before flight",
        "sms_pillar": SMSPillar.PROMOTION,
        "regulations": ["ICAO Annex 13"],
        "actions": [
            "Review similar incidents",
            "Brief crew on specific risks",
            "Implement threat and error management",
            "Consider additional safety briefing"
        ],
        "kpis": ["Incident recurrence rate", "Near-miss reports"]
    },
    
    "FATAL_RISK_ELEVATED": {
        "feature": "severity_code",
        "condition": lambda x: x >= 3,
        "title": "FATAL Risk Level - Comprehensive Review",
        "description": "Conditions match historical fatal accident patterns. Full risk review required.",
        "priority": Priority.CRITICAL,
        "risk_reduction": 50.0,
        "cost": CostLevel.LOW,
        "time": "Immediate",
        "sms_pillar": SMSPillar.RISK_MGMT,
        "regulations": ["ICAO SMS Manual"],
        "actions": [
            "Conduct pre-flight risk assessment",
            "Brief all crew on heightened awareness",
            "Establish hard risk mitigation criteria",
            "Document and share safety concern"
        ],
        "kpis": ["Pre-flight risk assessment completion", "Safety stand-down rate"]
    },
}


# ============================================
# STEP 4: Recommendation Engine Class
# ============================================

class RecommendationEngine:
    """
    Advanced recommendation engine that generates actionable safety recommendations
    based on risk factors, SHAP values, and SMS principles.
    """
    
    def __init__(self, model=None, explainer=None):
        self.model = model
        self.explainer = explainer
        self.rules = RECOMMENDATION_RULES
        print("   ✅ Recommendation Engine initialized")
        print(f"   Loaded {len(self.rules)} recommendation rules")
    
    def analyze(self, data: pd.DataFrame) -> List[Recommendation]:
        """
        Analyze data and generate recommendations.
        
        Args:
            data: DataFrame with operational/incident features
            
        Returns:
            List of Recommendation objects sorted by priority
        """
        recommendations = []
        
        # Get SHAP values if explainer available
        shap_values = None
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(data)
            except:
                pass
        
        # Check each rule
        for rule_id, rule in self.rules.items():
            feature = rule["feature"]
            
            # Skip if feature not in data
            if feature not in data.columns:
                continue
            
            # Get feature value(s)
            values = data[feature].values
            
            # Check if any value triggers the rule
            for idx, value in enumerate(values):
                if pd.isna(value):
                    continue
                    
                if rule["condition"](value):
                    # Get SHAP contribution if available
                    shap_contrib = 0.0
                    if shap_values is not None:
                        try:
                            feat_idx = list(data.columns).index(feature)
                            shap_contrib = float(shap_values[idx, feat_idx])
                        except:
                            pass
                    
                    rec = Recommendation(
                        id=f"{rule_id}_{idx}",
                        title=rule["title"],
                        description=rule["description"],
                        priority=rule["priority"].value,
                        risk_reduction_percent=rule["risk_reduction"],
                        cost_level=rule["cost"].value,
                        implementation_time=rule["time"],
                        sms_pillar=rule["sms_pillar"].value,
                        regulatory_refs=rule["regulations"],
                        shap_contribution=shap_contrib,
                        trigger_value=float(value),
                        trigger_threshold=self._get_threshold(rule["condition"]),
                        action_steps=rule["actions"],
                        kpis=rule["kpis"]
                    )
                    recommendations.append(rec)
        
        # Sort by priority (CRITICAL > HIGH > MEDIUM > LOW)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))
        
        return recommendations
    
    def _get_threshold(self, condition) -> float:
        """Extract threshold value from condition function."""
        # Try to get the comparison value from the condition
        # This is a simplified approach
        return 0.0  # Default if we can't extract
    
    def generate_report(self, recommendations: List[Recommendation]) -> str:
        """Generate a formatted safety recommendation report."""
        if not recommendations:
            return "No recommendations generated. All risk factors within acceptable limits."
        
        report = []
        report.append("=" * 70)
        report.append("📋 AERORISK SAFETY RECOMMENDATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Summary
        critical = sum(1 for r in recommendations if r.priority == "CRITICAL")
        high = sum(1 for r in recommendations if r.priority == "HIGH")
        medium = sum(1 for r in recommendations if r.priority == "MEDIUM")
        low = sum(1 for r in recommendations if r.priority == "LOW")
        
        report.append(f"\n📊 SUMMARY: {len(recommendations)} recommendations")
        report.append(f"   🔴 CRITICAL: {critical}")
        report.append(f"   🟠 HIGH:     {high}")
        report.append(f"   🟡 MEDIUM:   {medium}")
        report.append(f"   🟢 LOW:      {low}")
        
        # Total risk reduction potential
        total_reduction = sum(r.risk_reduction_percent for r in recommendations)
        report.append(f"\n   📉 Total Risk Reduction Potential: {min(total_reduction, 95):.0f}%")
        
        report.append("\n" + "-" * 70)
        report.append("DETAILED RECOMMENDATIONS")
        report.append("-" * 70)
        
        for i, rec in enumerate(recommendations, 1):
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec.priority, "⚪")
            
            report.append(f"\n{icon} [{rec.priority}] #{i}: {rec.title}")
            report.append(f"   {rec.description}")
            report.append(f"")
            report.append(f"   📈 Risk Reduction:    {rec.risk_reduction_percent:.0f}%")
            report.append(f"   💰 Cost:              {rec.cost_level}")
            report.append(f"   ⏱️  Implementation:    {rec.implementation_time}")
            report.append(f"   📋 SMS Pillar:        {rec.sms_pillar}")
            report.append(f"   🏛️  Regulations:       {', '.join(rec.regulatory_refs)}")
            
            if rec.shap_contribution != 0:
                report.append(f"   🔬 SHAP Contribution: {rec.shap_contribution:+.2f}")
            
            report.append(f"\n   📝 Action Steps:")
            for step in rec.action_steps:
                report.append(f"      • {step}")
            
            report.append(f"\n   📊 KPIs to Track:")
            for kpi in rec.kpis:
                report.append(f"      • {kpi}")
        
        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def to_json(self, recommendations: List[Recommendation]) -> str:
        """Export recommendations as JSON for API consumption."""
        return json.dumps([asdict(r) for r in recommendations], indent=2)
    
    def to_dataframe(self, recommendations: List[Recommendation]) -> pd.DataFrame:
        """Export recommendations as DataFrame."""
        return pd.DataFrame([asdict(r) for r in recommendations])


# ============================================
# STEP 5: Main Execution
# ============================================

if __name__ == "__main__":
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'recommendations')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize engine
    print("\n🔧 Initializing Recommendation Engine...")
    engine = RecommendationEngine()
    
    # Load sample data
    print("\n📥 Loading data for demonstration...")
    
    # Try operational data first
    ops_path = os.path.join(DATA_DIR, 'operational_features_latest.parquet')
    inc_path = os.path.join(DATA_DIR, 'incident_features_latest.parquet')
    
    if os.path.exists(ops_path):
        df = pd.read_parquet(ops_path)
        print(f"   ✅ Loaded {len(df):,} operational records")
    elif os.path.exists(inc_path):
        df = pd.read_parquet(inc_path)
        print(f"   ✅ Loaded {len(df):,} incident records")
    else:
        print("   ❌ No data found!")
        exit(1)
    
    # Analyze a sample of high-risk records
    print("\n🔍 Analyzing high-risk samples...")
    
    # Get some interesting samples (high risk or anomalous)
    if 'operational_risk_score' in df.columns:
        high_risk = df.nlargest(10, 'operational_risk_score')
    elif 'injury_score' in df.columns:
        high_risk = df.nlargest(10, 'injury_score')
    else:
        high_risk = df.sample(10, random_state=42)
    
    print(f"   Selected {len(high_risk)} high-risk samples for analysis")
    
    # Generate recommendations
    print("\n💡 Generating recommendations...")
    recommendations = engine.analyze(high_risk)
    print(f"   ✅ Generated {len(recommendations)} recommendations")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RECOMMENDATION SUMMARY")
    print("=" * 60)
    
    priority_counts = {}
    for rec in recommendations:
        priority_counts[rec.priority] = priority_counts.get(rec.priority, 0) + 1
    
    print(f"\n   Total Recommendations: {len(recommendations)}")
    for priority, count in sorted(priority_counts.items()):
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
        print(f"   {icon} {priority}: {count}")
    
    # Top 5 recommendations
    print("\n   Top 5 Recommendations:")
    print("   " + "-" * 50)
    for i, rec in enumerate(recommendations[:5], 1):
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec.priority, "⚪")
        print(f"   {i}. {icon} {rec.title}")
        print(f"      Risk Reduction: {rec.risk_reduction_percent:.0f}% | Cost: {rec.cost_level}")
    
    # Generate and save full report
    print("\n📝 Generating detailed report...")
    report = engine.generate_report(recommendations)
    
    report_path = os.path.join(OUTPUT_DIR, 'safety_recommendations.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   [OK] Report saved to: {report_path}")
    
    # Save as JSON (for API)
    json_path = os.path.join(OUTPUT_DIR, 'recommendations.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(engine.to_json(recommendations))
    print(f"   [OK] JSON saved to: {json_path}")
    
    # Save as CSV
    csv_path = os.path.join(OUTPUT_DIR, 'recommendations.csv')
    engine.to_dataframe(recommendations).to_csv(csv_path, index=False)
    print(f"   ✅ CSV saved to: {csv_path}")
    
    # Print sample of the report
    print("\n" + "=" * 60)
    print("📋 SAMPLE REPORT OUTPUT")
    print("=" * 60)
    print(report[:2000])
    if len(report) > 2000:
        print("\n... [truncated, see full report in file]")
    
    print("\n" + "=" * 60)
    print("✅ RECOMMENDATION ENGINE COMPLETE!")
    print("=" * 60)
    print(f"""
   Files generated:
   📁 {OUTPUT_DIR}/
      ├── safety_recommendations.txt  ← Full report
      ├── recommendations.json        ← API-ready
      └── recommendations.csv         ← Spreadsheet format
   
   Usage in production:
   
   from recommendation_engine import RecommendationEngine
   
   engine = RecommendationEngine()
   recommendations = engine.analyze(operational_data)
   report = engine.generate_report(recommendations)
""")
    print("=" * 60)
