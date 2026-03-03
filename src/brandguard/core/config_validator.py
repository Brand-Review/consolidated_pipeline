"""
Config-first enforcement layer.
Blocks analysis unless configuration is valid.
"""

from typing import Dict, Any


def evaluate_config_state(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    V1 RULES:
    - Block ONLY if BOTH brandName AND brandPurpose are missing
    - All other fields (logo/fonts/colors) are OPTIONAL
    - analysisMode = "compliance_enabled" if brandName AND brandPurpose exist
    - analysisMode = "observational_only" if either is missing
    """
    brand_name = (config.get("brandName") or "").strip()
    brand_purpose = (config.get("brandPurpose") or config.get("industry") or "").strip()
    
    has_brand_name = bool(brand_name)
    has_purpose = bool(brand_purpose)
    
    # V1 RULE: Block ONLY if BOTH are missing
    if not has_brand_name and not has_purpose:
        config_state = "not_configured"
        analysis_mode = "observational_only"
    elif has_brand_name and has_purpose:
        # V1 RULE: Both exist → compliance enabled (even without logo/fonts/colors)
        config_state = "ready"
        analysis_mode = "compliance_enabled"
    else:
        # V1 RULE: One missing → partial, but still allow upload
        config_state = "partial"
        analysis_mode = "observational_only"
    
    return {
        "configState": config_state,
        "analysisMode": analysis_mode
    }


def build_block_response(config_state: str) -> Dict[str, Any]:
    """
    V1 RULE: Only called when BOTH brandName AND brandPurpose are missing
    """
    return {
        "success": False,
        "errorType": "CONFIG_REQUIRED",
        "message": "Please complete brand setup: Brand Name and Brand Purpose are required.",
        "requiredActions": [
            "Add brand name",
            "Add brand purpose"
        ],
        "configState": config_state,
        "redirectTo": "/setup"
    }

