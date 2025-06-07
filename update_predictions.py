import os
from google.cloud import storage
import json
from pathlib import Path

def update_predictions_from_gcs():
    """
    Updates prediction JSONs in the predictions folder by downloading them from
    the Google Cloud Storage bucket 'sp-500-tech-analysis'
    """
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket_name = 'sp-500-tech-analysis'
    bucket = storage_client.bucket(bucket_name)
    
    # Create predictions directory if it doesn't exist
    predictions_dir = Path('predictions')
    predictions_dir.mkdir(exist_ok=True)
    
    # List all blobs (files) in the bucket
    blobs = bucket.list_blobs()
    
    # Download each JSON file
    for blob in blobs:
        if blob.name.endswith('.json'):
            # Get the filename from the blob name
            filename = os.path.basename(blob.name)
            local_path = predictions_dir / filename
            
            # Download the file
            blob.download_to_filename(local_path)
            print(f"Downloaded {filename} to {local_path}")

if __name__ == "__main__":
    update_predictions_from_gcs() 