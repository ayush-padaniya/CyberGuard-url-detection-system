import boto3
import os
from src.logger import logging

# ==============================================
#         S3 Connection Utility
# ==============================================


def get_s3_client():
    """Create and return S3 client using environment variables."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        logging.info("S3 client created successfully")
        return s3_client
    except Exception as e:
        logging.error(f"Error creating S3 client: {e}")
        raise

# ==============================================
#             Downloads Data From S3
# ==============================================

def download_file_from_s3(bucket_name: str, s3_key: str, local_path: str):
    """Download file from S3 to local path."""
    try:
        # initialize S3 client
        s3_client = get_s3_client()

        # download file from S3
        s3_client.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Downloaded {s3_key} from S3 to {local_path}")

    except Exception as e:
        logging.error(f"Error downloading from S3: {e}")
        raise


# ==============================================
#             Uploads Data To S3
# ==============================================

def upload_file_to_s3(local_path: str, bucket_name: str, s3_key: str):
    """Upload file from local path to S3."""
    try:
        # initialize S3 client
        s3_client = get_s3_client()

        # upload file to S3
        s3_client.upload_file(local_path, bucket_name, s3_key)
        logging.info(f"Uploaded {local_path} to S3 as {s3_key}")

    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")
        raise