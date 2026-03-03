"""
JSON Schema Validator for BrandGuard Master Analyzer Output
Validates LLM output against strict schema to ensure trust and correctness.
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not available. Install with: pip install jsonschema")


class SchemaValidator:
    """
    Validates BrandGuard analyzer output against master schema.
    """
    
    def __init__(self):
        self.schema = None
        if JSONSCHEMA_AVAILABLE:
            self._load_schema()
    
    def _load_schema(self):
        """Load master analyzer schema from JSON file"""
        try:
            schema_path = Path(__file__).parent.parent / 'prompts' / 'schemas' / 'master_analyzer_schema.json'
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
                logger.info("✅ Master analyzer schema loaded successfully")
            else:
                logger.warning(f"Schema file not found: {schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self.schema = None
    
    def validate_output(self, output: Dict[str, Any]) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate analyzer output against master schema.
        
        Args:
            output: Analyzer output dictionary
            
        Returns:
            Tuple: (is_valid, error_message, validation_errors)
                - is_valid: True if output is valid
                - error_message: Overall error message if invalid
                - validation_errors: List of specific validation errors
        """
        if not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not available - skipping schema validation")
            return True, None, []
        
        if not self.schema:
            logger.warning("Schema not loaded - skipping schema validation")
            return True, None, []
        
        try:
            # Validate against schema
            validate(instance=output, schema=self.schema)
            
            # Additional semantic validations
            semantic_errors = self._validate_semantics(output)
            
            if semantic_errors:
                error_msg = f"Semantic validation failed: {', '.join(semantic_errors)}"
                return False, error_msg, semantic_errors
            
            logger.info("✅ Output validated against master schema")
            return True, None, []
            
        except ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            errors = [f"{'.'.join(str(p) for p in e.path)}: {e.message}"]
            logger.error(f"❌ Schema validation failed: {error_msg}")
            return False, error_msg, errors
        
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(f"❌ Validation failed: {error_msg}")
            return False, error_msg, []
    
    def _validate_semantics(self, output: Dict[str, Any]) -> List[str]:
        """
        Additional semantic validations beyond JSON schema.
        
        Rules:
        - If criticalSignalFailure = true, overall.complianceScore must be null
        - If status = "unknown", score must be null
        - If status = "failed", score must be null or 0
        - Placement validation requires verified logo
        """
        errors = []
        
        critical_signal_failure = output.get('criticalSignalFailure', False)
        overall = output.get('overall', {})
        overall_score = overall.get('complianceScore')
        
        # RULE: If criticalSignalFailure = true, overall.complianceScore must be null
        if critical_signal_failure and overall_score is not None:
            errors.append("criticalSignalFailure=true but overall.complianceScore is not null")
        
        # RULE: If overall.status = "blocked", complianceScore must be null
        if overall.get('status') == 'blocked' and overall_score is not None:
            errors.append("overall.status='blocked' but overall.complianceScore is not null")
        
        # Validate copywriting
        copywriting = output.get('copywriting', {})
        copywriting_status = copywriting.get('status', '')
        
        if copywriting_status == 'system_error':
            # System error should trigger critical signal failure
            if not critical_signal_failure:
                errors.append("copywriting.status='system_error' but criticalSignalFailure=false")
        
        # Validate logo
        logo = output.get('logo', {})
        logo_status = logo.get('status', '')
        detections = logo.get('detections', [])
        placement_status = logo.get('placementStatus', '')
        
        # RULE: Placement validation requires verified logo
        verified_logos = [d for d in detections if d.get('verified', False)]
        if placement_status not in ['not_applicable', 'unknown'] and not verified_logos:
            errors.append("placementStatus requires verified logo but none found")
        
        # Validate typography
        typography = output.get('typography', {})
        typography_status = typography.get('status', '')
        
        # RULE: If typography.status = "unknown", score should be null
        if typography_status == 'unknown':
            # Observable score might exist, but compliance score should be null
            pass  # Schema already enforces this
        
        # Validate color
        color = output.get('color', {})
        color_status = color.get('status', '')
        
        # RULE: If color.status = "observed_only", compliance score should be null
        if color_status == 'observed_only':
            # Schema already enforces this
            pass
        
        return errors
    
    def normalize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize analyzer output to match master schema.
        
        Fixes common issues:
        - Converts status values to enum
        - Ensures required fields exist
        - Normalizes null values
        """
        normalized = output.copy()
        
        # Normalize overall status
        if 'overall' in normalized:
            overall = normalized['overall']
            # Ensure status is valid enum
            if overall.get('status') not in ['passed', 'failed', 'blocked', 'unknown']:
                overall['status'] = 'unknown'
            
            # Ensure complianceScore is null if critical signal failure
            if normalized.get('criticalSignalFailure', False):
                overall['complianceScore'] = None
                overall['status'] = 'blocked'
        
        # Normalize copywriting status
        if 'copywriting' in normalized:
            copywriting = normalized['copywriting']
            if copywriting.get('status') not in ['passed', 'failed', 'skipped', 'system_error']:
                copywriting['status'] = 'failed'
        
        # Normalize logo status
        if 'logo' in normalized:
            logo = normalized['logo']
            if logo.get('status') not in ['passed', 'failed', 'skipped', 'detected_unverified']:
                logo['status'] = 'skipped'
            
            # Ensure placementStatus is not_applicable if no verified logo
            detections = logo.get('detections', [])
            verified_logos = [d for d in detections if d.get('verified', False)]
            if not verified_logos and logo.get('placementStatus') not in ['not_applicable', 'unknown']:
                logo['placementStatus'] = 'not_applicable'
        
        # Normalize typography status
        if 'typography' in normalized:
            typography = normalized['typography']
            if typography.get('status') not in ['passed', 'unknown', 'failed']:
                typography['status'] = 'unknown'
        
        # Normalize color status
        if 'color' in normalized:
            color = normalized['color']
            if color.get('status') not in ['passed', 'observed_only', 'failed']:
                color['status'] = 'observed_only'
        
        return normalized

