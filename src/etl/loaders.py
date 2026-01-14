"""
AeroRisk - Data Loaders
Load transformed data back to database
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import text
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.connection import engine, get_db_session


class DataLoader:
    """Load transformed data to PostgreSQL."""
    
    def __init__(self, chunk_size: int = 5000):
        self.chunk_size = chunk_size
        self.engine = engine
    
    def load_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str,
        if_exists: str = 'append',
        index: bool = False
    ) -> int:
        """
        Load DataFrame to database table.
        
        Args:
            df: DataFrame to load
            table_name: Target table name
            schema: Database schema
            if_exists: 'append', 'replace', or 'fail'
            index: Whether to include DataFrame index
            
        Returns:
            Number of records loaded
        """
        logger.info(f"Loading {len(df):,} records to {schema}.{table_name}...")
        
        total_loaded = 0
        
        try:
            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i:i + self.chunk_size]
                chunk.to_sql(
                    table_name,
                    self.engine,
                    schema=schema,
                    if_exists=if_exists if i == 0 else 'append',
                    index=index,
                    method='multi'
                )
                total_loaded += len(chunk)
                
                if total_loaded % 10000 == 0 or total_loaded == len(df):
                    logger.info(f"  Loaded {total_loaded:,}/{len(df):,} records")
            
            logger.info(f"✅ Successfully loaded {total_loaded:,} records to {schema}.{table_name}")
            return total_loaded
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def upsert_predictions(self, predictions_df: pd.DataFrame) -> int:
        """
        Upsert risk predictions (insert or update existing).
        
        Args:
            predictions_df: DataFrame with prediction data
            
        Returns:
            Number of records upserted
        """
        logger.info(f"Upserting {len(predictions_df):,} predictions...")
        
        # For now, just append - in production would use PostgreSQL UPSERT
        return self.load_to_table(
            predictions_df,
            'risk_predictions',
            'ml',
            if_exists='append'
        )
    
    def log_data_quality(self, report_dict: Dict) -> None:
        """
        Log data quality report to database.
        
        Args:
            report_dict: Data quality report as dictionary
        """
        logger.info(f"Logging data quality report for {report_dict['source']}...")
        
        quality_record = pd.DataFrame([{
            'run_date': datetime.now(),
            'source': report_dict['source'],
            'total_records': report_dict['total_records'],
            'valid_records': int(report_dict['total_records'] * report_dict['overall_score'] / 100),
            'invalid_records': int(report_dict['total_records'] * (1 - report_dict['overall_score'] / 100)),
            'completeness_score': report_dict['completeness_score'],
            'consistency_score': report_dict['consistency_score'],
            'accuracy_score': report_dict['accuracy_score'],
            'overall_score': report_dict['overall_score'],
            'validation_errors': None,
            'recommendations': None
        }])
        
        self.load_to_table(
            quality_record,
            'data_quality_logs',
            'analytics',
            if_exists='append'
        )
        
        logger.info("✅ Data quality report logged")


class FeatureStore:
    """Store and retrieve ML features."""
    
    def __init__(self):
        self.engine = engine
    
    def save_features(
        self,
        features_df: pd.DataFrame,
        feature_set_name: str
    ) -> int:
        """
        Save feature set to feature store.
        
        Args:
            features_df: DataFrame with computed features
            feature_set_name: Name for this feature set
            
        Returns:
            Number of features saved
        """
        logger.info(f"Saving feature set '{feature_set_name}' ({len(features_df):,} records)...")
        
        # Create a copy to avoid modifying original
        df_to_save = features_df.copy()
        
        # Add metadata
        df_to_save['feature_set_name'] = feature_set_name
        df_to_save['created_at'] = datetime.now()
        
        # Convert UUID columns to strings (parquet doesn't support UUID natively)
        for col in df_to_save.columns:
            if df_to_save[col].dtype == 'object':
                # Check if column contains UUID objects
                sample = df_to_save[col].dropna().head(1)
                if len(sample) > 0:
                    import uuid
                    if isinstance(sample.iloc[0], uuid.UUID):
                        df_to_save[col] = df_to_save[col].astype(str)
        
        # Convert any remaining object columns that might cause issues
        for col in df_to_save.select_dtypes(include=['object']).columns:
            try:
                df_to_save[col] = df_to_save[col].astype(str)
            except:
                pass
        
        # Save to data directory as parquet for ML training
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', f'{feature_set_name}.parquet'
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_to_save.to_parquet(output_path, index=False)
        
        logger.info(f"✅ Saved {len(df_to_save):,} features to {output_path}")
        return len(df_to_save)
    
    def load_features(self, feature_set_name: str) -> Optional[pd.DataFrame]:
        """
        Load feature set from feature store.
        
        Args:
            feature_set_name: Name of feature set to load
            
        Returns:
            DataFrame with features or None if not found
        """
        feature_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', f'{feature_set_name}.parquet'
        )
        
        if os.path.exists(feature_path):
            logger.info(f"Loading feature set '{feature_set_name}'...")
            return pd.read_parquet(feature_path)
        
        logger.warning(f"Feature set '{feature_set_name}' not found")
        return None
    
    def list_feature_sets(self) -> List[str]:
        """List available feature sets."""
        processed_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed'
        )
        
        if not os.path.exists(processed_dir):
            return []
        
        return [
            f.replace('.parquet', '') 
            for f in os.listdir(processed_dir) 
            if f.endswith('.parquet')
        ]
