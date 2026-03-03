"""
Google Cloud Vision Credentials Validator
Validates Google Cloud Vision API credentials at startup.
Raises hard errors if credentials are missing or invalid.
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_google_credentials(raise_on_error: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate Google Cloud Vision API credentials at startup.
    
    Args:
        raise_on_error: If True, raises RuntimeError on validation failure. If False, returns (False, error_message).
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if credentials are valid, False otherwise
        - error_message: Error message if validation failed, None if valid
    
    Raises:
        RuntimeError: If raise_on_error=True and credentials are invalid
    """
    try:
        # Check if google-cloud-vision is installed
        try:
            from google.cloud import vision
            from google.oauth2 import service_account
        except ImportError:
            error_msg = (
                "❌ Google Cloud Vision library not installed.\n"
                "   Install with: pip install google-cloud-vision>=3.7.2\n"
                "   Google OCR will not function without this library."
            )
            if raise_on_error:
                raise RuntimeError(error_msg)
            return False, error_msg
        
        # Try to get credentials from settings first
        credentials_path = None
        try:
            from ..config.settings import settings
            credentials_path = settings.google_application_credentials
        except (ImportError, AttributeError):
            pass
        
        # If settings not available, try reading directly from config file
        if not credentials_path:
            try:
                import yaml
                # Try to find production.yaml config file
                config_paths = [
                    Path('configs/production.yaml'),
                    Path(__file__).parent.parent.parent.parent / 'configs' / 'production.yaml',
                    Path.cwd() / 'configs' / 'production.yaml',
                ]
                
                for config_path in config_paths:
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        
                        # Check both top-level and nested under api_credentials
                        if 'google_application_credentials' in config_data:
                            credentials_path = config_data['google_application_credentials']
                        elif 'api_credentials' in config_data and config_data['api_credentials']:
                            credentials_path = config_data['api_credentials'].get('google_application_credentials')
                        
                        if credentials_path:
                            break
            except Exception:
                pass
        
        # Fallback to environment variable
        if not credentials_path:
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Check if credentials path is set
        if not credentials_path:
            error_msg = (
                "❌ GOOGLE_APPLICATION_CREDENTIALS not set.\n"
                "   Set it in configs/production.yaml:\n"
                "     api_credentials:\n"
                "       google_application_credentials: \"/path/to/service-account.json\"\n"
                "   OR set environment variable:\n"
                "     export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\"\n"
                "   Google OCR will not function without credentials."
            )
            if raise_on_error:
                raise RuntimeError(error_msg)
            return False, error_msg
        
        # Expand user path (~) if present
        credentials_path = os.path.expanduser(credentials_path)
        
        # Check if credentials file exists
        if not os.path.exists(credentials_path):
            error_msg = (
                f"❌ Google credentials file not found: {credentials_path}\n"
                "   Please check the path in configs/production.yaml or GOOGLE_APPLICATION_CREDENTIALS environment variable.\n"
                "   Google OCR will not function without valid credentials."
            )
            if raise_on_error:
                raise RuntimeError(error_msg)
            return False, error_msg
        
        # Validate JSON file format
        try:
            import json
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
            
            # Check if it's a valid service account JSON
            if not isinstance(creds_data, dict):
                error_msg = f"❌ Invalid credentials file format: {credentials_path} (not a JSON object)"
                if raise_on_error:
                    raise RuntimeError(error_msg)
                return False, error_msg
            
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in creds_data]
            if missing_fields:
                error_msg = (
                    f"❌ Invalid service account JSON: {credentials_path}\n"
                    f"   Missing required fields: {', '.join(missing_fields)}\n"
                    "   Please ensure you downloaded the correct service account key from Google Cloud Console."
                )
                if raise_on_error:
                    raise RuntimeError(error_msg)
                return False, error_msg
            
            if creds_data.get('type') != 'service_account':
                error_msg = (
                    f"❌ Invalid credentials type: {credentials_path}\n"
                    "   Expected 'service_account' type. Please ensure you downloaded a service account key."
                )
                if raise_on_error:
                    raise RuntimeError(error_msg)
                return False, error_msg
            
        except json.JSONDecodeError as e:
            error_msg = (
                f"❌ Invalid JSON in credentials file: {credentials_path}\n"
                f"   Error: {str(e)}\n"
                "   Please ensure the file is valid JSON."
            )
            if raise_on_error:
                raise RuntimeError(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = (
                f"❌ Error reading credentials file: {credentials_path}\n"
                f"   Error: {str(e)}"
            )
            if raise_on_error:
                raise RuntimeError(error_msg)
            return False, error_msg
        
        # Try to initialize the client to validate credentials
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = vision.ImageAnnotatorClient(credentials=credentials)
            
            # Validate that we can create the client (this validates the credentials format)
            # Note: We don't make an actual API call here to avoid network delays at startup
            # The credentials will be validated when first used
            logger.info(f"✅ Google Cloud Vision credentials validated: {credentials_path}")
            logger.info(f"   Project ID: {creds_data.get('project_id', 'unknown')}")
            logger.info(f"   Service Account: {creds_data.get('client_email', 'unknown')}")
            
            return True, None
            
        except Exception as e:
            error_msg = (
                f"❌ Failed to initialize Google Cloud Vision client: {str(e)}\n"
                f"   Credentials file: {credentials_path}\n"
                "   Please verify:\n"
                "   1. The service account has 'Cloud Vision API User' role\n"
                "   2. Cloud Vision API is enabled in your Google Cloud project\n"
                "   3. The credentials file is not corrupted"
            )
            if raise_on_error:
                raise RuntimeError(error_msg)
            return False, error_msg
            
    except RuntimeError:
        # Re-raise RuntimeError (from validation failures above)
        raise
    except Exception as e:
        error_msg = f"❌ Unexpected error validating Google credentials: {str(e)}"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return False, error_msg

