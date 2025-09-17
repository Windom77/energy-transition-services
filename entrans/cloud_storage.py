import os
from google.cloud import storage
from pathlib import Path
import json
from datetime import datetime

class EnTransStorage:
    def __init__(self):
        self.client = storage.Client()
        self.data_bucket = self.client.bucket('entrans-467104-data')
        self.uploads_bucket = self.client.bucket('entrans-467104-uploads')
    
    def save_analysis_results(self, session_id, results_data):
        """Save analysis results to persistent storage"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"results/{session_id}_{timestamp}.json"
        
        blob = self.data_bucket.blob(file_path)
        blob.upload_from_string(
            json.dumps(results_data, indent=2),
            content_type='application/json'
        )
        return file_path
    
    def save_results_files(self, session_id, files_dict):
        """Save parquet/csv results files"""
        saved_files = {}
        for filename, file_content in files_dict.items():
            file_path = f"results/{session_id}/{filename}"
            blob = self.data_bucket.blob(file_path)
            
            if isinstance(file_content, bytes):
                blob.upload_from_string(file_content)
            else:
                blob.upload_from_string(str(file_content))
            saved_files[filename] = file_path
        return saved_files
    
    def load_analysis_results(self, session_id):
        """Load the most recent analysis results for a session"""
        prefix = f"results/{session_id}_"
        blobs = list(self.data_bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            return None
            
        # Get most recent
        latest_blob = max(blobs, key=lambda b: b.time_created)
        return json.loads(latest_blob.download_as_text())
    
    def cache_weather_data(self, location, weather_data):
        """Cache weather data to avoid re-downloading"""
        file_path = f"cache/weather/{location}.json"
        blob = self.data_bucket.blob(file_path)
        blob.upload_from_string(
            json.dumps(weather_data),
            content_type='application/json'
        )
    
    def get_cached_weather(self, location):
        """Retrieve cached weather data"""
        file_path = f"cache/weather/{location}.json"
        blob = self.data_bucket.blob(file_path)
        
        try:
            return json.loads(blob.download_as_text())
        except:
            return None

# Global storage instance
try:
    storage_manager = EnTransStorage()
    STORAGE_AVAILABLE = True
    print("✅ Cloud Storage initialized")
except Exception as e:
    print(f"⚠️ Cloud Storage not available: {e}")
    STORAGE_AVAILABLE = False
    storage_manager = None
