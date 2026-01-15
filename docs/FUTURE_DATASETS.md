# 📊 AeroRisk - Future Datasets Roadmap

This document tracks additional datasets that can be integrated to enhance the AeroRisk platform.

---

## ✅ Currently Integrated

| Dataset | Source | Records | Status |
|---------|--------|---------|--------|
| NTSB Aviation Accidents | ntsb.gov | ~88K | ✅ Ingested |
| ASRS Incident Reports | asrs.arc.nasa.gov | ~200K | ✅ Ingested |
| Synthetic Weather Data | Generated | ~33K | ✅ Merged |

---

## 🔄 Priority Datasets (Next Phase)

### 1. FAA Wildlife Strikes Database
- **Source:** https://wildlife.faa.gov/
- **Records:** ~300,000+
- **Value:** Bird strike risk analysis, seasonal patterns
- **Fields:** Species, airport, damage level, aircraft type, time of day
- **Effort:** Medium

### 2. FAA Service Difficulty Reports (SDR)
- **Source:** https://av-info.faa.gov/sdrx/
- **Records:** ~500,000+
- **Value:** Maintenance failure patterns, component reliability
- **Fields:** Part number, failure type, aircraft model, operator
- **Effort:** Medium

### 3. NOAA Aviation Weather Data
- **Source:** https://www.aviationweather.gov/
- **Records:** Continuous real-time
- **Value:** Real weather integration, METAR/TAF data
- **Fields:** Wind, visibility, ceiling, turbulence, icing
- **Effort:** High (API integration)

### 4. FAA Operations Network (OPSNET)
- **Source:** FAA
- **Records:** Millions
- **Value:** Traffic density, delay patterns
- **Fields:** Airport operations, delays, cancellations
- **Effort:** High (requires access)

---

## 📈 Enhancement Datasets

### 5. Flight Aware Historical Data
- **Source:** FlightAware API
- **Value:** Route-specific risk patterns
- **Cost:** Commercial API

### 6. OpenSky Network
- **Source:** https://opensky-network.org/
- **Value:** Real-time flight tracking, trajectory analysis
- **Cost:** Free for research

### 7. EUROCONTROL Safety Data
- **Source:** eurocontrol.int
- **Value:** European aviation safety patterns
- **Effort:** Medium

### 8. Aircraft Registration Database
- **Source:** FAA N-Number Registry
- **Value:** Aircraft age, model, owner tracking
- **Effort:** Low

---

## 🛠️ Implementation Notes

### Data Ingestion Pattern
```python
# Use existing ETL pattern in src/etl/
# 1. Create fetcher in src/ingestion/
# 2. Add transformer in src/etl/transformers.py
# 3. Update database schema if needed
# 4. Run ingestion pipeline
```

### Database Schema Extensions
```sql
-- Example for wildlife strikes
CREATE TABLE ingestion.wildlife_strikes (
    id SERIAL PRIMARY KEY,
    incident_date DATE,
    airport_code VARCHAR(10),
    species VARCHAR(100),
    damage_level VARCHAR(20),
    aircraft_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 📅 Suggested Timeline

| Phase | Datasets | Timeframe |
|-------|----------|-----------|
| Phase 8 | Wildlife Strikes, SDR | 2-3 weeks |
| Phase 9 | NOAA Weather API | 3-4 weeks |
| Phase 10 | OpenSky Integration | 2-3 weeks |

---

## 📝 Notes

- All datasets should follow the existing schema pattern in `ingestion` schema
- Weather data should be real-time for production use
- Consider data licensing/terms of use before commercial deployment
- Prioritize datasets that improve prediction accuracy

---

*Last Updated: January 2026*
