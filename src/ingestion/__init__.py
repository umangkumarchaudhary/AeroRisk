"""
AeroRisk Ingestion Module

Data fetchers for various aviation safety data sources.
"""

from src.ingestion.ntsb_fetcher import NTSBFetcher

__all__ = [
    "NTSBFetcher",
]
