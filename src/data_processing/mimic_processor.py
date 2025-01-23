import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

class MIMICDataProcessor:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_mimic_tables(self):
        """Load relevant MIMIC-III tables."""
        try:
            self.logger.info("Loading MIMIC-III tables...")
            
            # Load core tables with specific columns we need
            admissions = pd.read_csv(self.data_path / 'ADMISSIONS.csv', parse_dates=[
                'admittime', 'dischtime', 'edregtime', 'edouttime'
            ])
            
            transfers = pd.read_csv(self.data_path / 'TRANSFERS.csv', parse_dates=[
                'intime', 'outtime'
            ])
            
            patients = pd.read_csv(self.data_path / 'PATIENTS.csv')
            
            icustays = pd.read_csv(self.data_path / 'ICUSTAYS.csv', parse_dates=[
                'intime', 'outtime'
            ])
            
            services = pd.read_csv(self.data_path / 'SERVICES.csv', parse_dates=[
                'transfertime'
            ])

            return admissions, transfers, patients, icustays, services

        except Exception as e:
            self.logger.error(f"Error loading MIMIC tables: {str(e)}")
            raise

    def calculate_wait_times(self, admissions):
        """Calculate ED wait times from admission data."""
        try:
            # Calculate ED wait time
            admissions['ed_wait_time'] = (
                admissions['edouttime'] - admissions['edregtime']
            ).dt.total_seconds() / 60  # Convert to minutes

            # Add admission duration
            admissions['admission_duration'] = (
                admissions['dischtime'] - admissions['admittime']
            ).dt.total_seconds() / 3600  # Convert to hours

            # Group by admission_type
            wait_time_by_type = admissions.groupby('admission_type').agg({
                'ed_wait_time': ['mean', 'median', 'count'],
                'admission_duration': ['mean', 'median']
            }).round(2)

            # Flatten multi-level columns
            wait_time_by_type.columns = [
                'ed_wait_time_mean', 'ed_wait_time_median', 'ed_wait_time_count',
                'admission_duration_mean', 'admission_duration_median'
            ]
            wait_time_by_type = wait_time_by_type.reset_index()

            # Save the cleaned data permanently
            wait_time_by_type.to_csv('data/processed/metrics_wait_times.csv', index=False)

            return admissions, wait_time_by_type

        except Exception as e:
            self.logger.error(f"Error calculating wait times: {str(e)}")
            raise

    def process_transfers(self, transfers, icustays):
        """Process transfer data to analyze patient flow."""
        try:
            # Merge transfers with ICU stays to get more context
            transfers_icu = transfers.merge(
                icustays[['subject_id', 'hadm_id', 'icustay_id', 'first_careunit', 'last_careunit']],
                on=['subject_id', 'hadm_id', 'icustay_id'],
                how='left'
            )
            
            # Calculate length of stay
            transfers_icu['length_of_stay'] = (
                transfers_icu['outtime'] - transfers_icu['intime']
            ).dt.total_seconds() / 3600  # Convert to hours

            # Create ward flow metrics
            ward_metrics = transfers_icu.groupby('curr_wardid').agg({
                'subject_id': 'count',  # Number of patients
                'length_of_stay': ['mean', 'median', 'std']
            }).round(2)

            # Flatten multi-level columns
            ward_metrics.columns = [
                'subject_id_count', 'length_of_stay_mean',
                'length_of_stay_median', 'length_of_stay_std'
            ]
            ward_metrics = ward_metrics.reset_index()

            # Drop rows with missing ward IDs
            ward_metrics = ward_metrics.dropna(subset=['curr_wardid'])
            ward_metrics['curr_wardid'] = pd.to_numeric(ward_metrics['curr_wardid'], errors='coerce')
            ward_metrics = ward_metrics.dropna(subset=['curr_wardid'])

            # Save the cleaned data permanently
            ward_metrics.to_csv('data/processed/metrics_ward_metrics.csv', index=False)

            return transfers_icu, ward_metrics

        except Exception as e:
            self.logger.error(f"Error processing transfers: {str(e)}")
            raise

    def create_hourly_stats(self, admissions, transfers):
        """Create hourly statistics for dashboard."""
        try:
            # Hourly admission patterns
            admissions['hour'] = admissions['admittime'].dt.hour
            hourly_admissions = admissions.groupby(['hour', 'admission_type']).size().reset_index(name='admission_count')
            
            # Hourly transfer patterns
            transfers['hour'] = transfers['intime'].dt.hour
            hourly_transfers = transfers.groupby(['hour', 'curr_wardid']).size().reset_index(name='transfer_count')

            return hourly_admissions, hourly_transfers

        except Exception as e:
            self.logger.error(f"Error creating hourly stats: {str(e)}")
            raise

    def process_data(self):
        """Main processing pipeline."""
        try:
            # Load data
            admissions, transfers, patients, icustays, services = self.load_mimic_tables()
            
            # Process wait times
            admissions_processed, wait_time_metrics = self.calculate_wait_times(admissions)
            
            # Process transfers
            transfers_processed, ward_metrics = self.process_transfers(transfers, icustays)
            
            # Create hourly statistics
            hourly_admissions, hourly_transfers = self.create_hourly_stats(
                admissions_processed, transfers_processed
            )
            
            # Create final processed dataset
            processed_data = {
                'admissions': admissions_processed,
                'transfers': transfers_processed,
                'hourly_stats': {
                    'admissions': hourly_admissions,
                    'transfers': hourly_transfers
                },
                'metrics': {
                    'wait_times': wait_time_metrics,
                    'ward_metrics': ward_metrics
                }
            }

            return processed_data

        except Exception as e:
            self.logger.error(f"Error in main processing pipeline: {str(e)}")
            raise