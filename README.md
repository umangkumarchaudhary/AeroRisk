# 🛫 AeroRisk - Predictive Safety Risk Analytics Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An SMS-aligned predictive safety analytics platform that identifies and mitigates operational safety risks using historical incident, weather, and operational data.

## 🎯 Overview

AeroRisk is a comprehensive safety analytics platform designed to support aviation Safety Management Systems (SMS). It provides:

- **Predictive Analytics**: ML models predicting incident risk scores and severity
- **Prescriptive Insights**: Actionable recommendations with ROI analysis
- **SMS Alignment**: Full compliance with SMS pillars and safety performance indicators
- **Real-time Dashboards**: Interactive visualization of safety metrics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                        │
├─────────────────────────────────────────────────────┤
│ NTSB API │ ASRS │ Weather API │ Synthetic Ops Data │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              INGESTION & ETL (Prefect)              │
│  • Data validation  • Deduplication  • Enrichment   │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│           DATA WAREHOUSE (PostgreSQL)               │
│  • Staging  • Transformed  • Analytics-ready        │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│           ANALYTICS ENGINE (Python)                 │
│  ┌─────────────────┐  ┌──────────────────┐        │
│  │ Predictive      │  │ Prescriptive     │        │
│  │ Models          │  │ Recommendation   │        │
│  │ (XGBoost/RF)    │  │ Engine           │        │
│  └─────────────────┘  └──────────────────┘        │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│        API LAYER (FastAPI) + DASHBOARDS             │
│  • REST endpoints  • Streamlit  • Power BI          │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aerorisk.git
cd aerorisk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start PostgreSQL with Docker
docker-compose up -d postgres adminer

# Run database migrations
alembic upgrade head

# Verify database setup
python scripts/verify_db.py
```

### Access Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| Dashboard | http://localhost:8501 | Streamlit dashboard |
| Adminer | http://localhost:8080 | Database management |

## 🗃️ Database Schema

### Schemas

| Schema | Purpose |
|--------|---------|
| `ingestion` | Raw and processed data from external sources |
| `ml` | ML model predictions and registry |
| `analytics` | KPIs, recommendations, and audit logs |

### Core Tables

| Table | Schema | Description |
|-------|--------|-------------|
| `incidents` | ingestion | Historical incident records |
| `weather_conditions` | ingestion | Weather data by airport |
| `operational_data` | ingestion | Synthetic operational data |
| `risk_predictions` | ml | Model predictions |
| `model_registry` | ml | ML model versioning |
| `recommendations` | analytics | Prescriptive recommendations |
| `safety_kpis` | analytics | SMS performance indicators |

## 📊 Features

### Predictive Analytics
- Risk score prediction (0-100)
- Severity classification (None/Minor/Serious/Fatal)
- 87%+ accuracy target

### Prescriptive Engine
- Actionable mitigation recommendations
- ROI and cost-benefit analysis
- Priority-ranked actions

### SMS Alignment
- Safety Performance Indicators (SPIs)
- Safety Performance Targets (SPTs)
- Four pillars alignment

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data** | Pandas, Polars, NumPy |
| **Database** | PostgreSQL 16, SQLAlchemy 2.0 |
| **ML** | XGBoost, LightGBM, Scikit-learn, SHAP |
| **API** | FastAPI, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Pipeline** | Prefect |
| **Deployment** | Docker, Docker Compose |

## 📁 Project Structure

```
aerorisk/
├── src/
│   ├── ingestion/      # Data fetchers
│   ├── etl/            # Transformers & validators
│   ├── models/         # ML models
│   ├── analytics/      # KPI & recommendation engine
│   ├── api/            # FastAPI backend
│   └── database/       # SQLAlchemy models & migrations
├── dashboard/          # Streamlit app
├── pipelines/          # Prefect DAGs
├── tests/              # Test suite
├── notebooks/          # Jupyter notebooks
├── data/               # Data directories
├── models/             # Trained model artifacts
└── docs/               # Documentation
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- NTSB Aviation Accident Database
- NASA ASRS
- OpenWeather API
