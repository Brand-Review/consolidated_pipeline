"""
Pre-Scoring Guard Middleware
Enforces non-negotiable rules before scoring occurs.

This ensures:
- Never score unknown or failed signals
- Critical errors block overall scoring
- Proper status reporting
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PreScoringGuard:
    """
    Pre-scoring guard middleware that enforces non-negotiable rules.
    
    Rules:
    1. Never score unknown or failed signals
    2. OCR failure with visible text = critical error
    3. Logo detection failure ≠ no logo detected
    4. Typography compliance requires brand fonts
    5. Overall score must be null if any critical signal fails
    """
    
    def __init__(self):
        self.critical_errors = []
        self.blocked_signals = []
    
    def validate_analysis_results(self, analysis_results: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate analysis results before scoring.
        
        Args:
            analysis_results: Raw analysis results from analyzers
            
        Returns:
            Tuple: (is_valid, errors, validated_results)
                - is_valid: True if scoring can proceed
                - errors: List of error messages
                - validated_results: Validated results with enforced rules
        """
        self.critical_errors = []
        self.blocked_signals = []
        
        validated = {}
        
        # Validate each analyzer result
        validated['copywriting'], copy_errors = self._validate_copywriting(analysis_results.get('copywriting_analysis', {}))
        validated['logo'], logo_errors = self._validate_logo(analysis_results.get('logo_analysis', {}))
        validated['typography'], typo_errors = self._validate_typography(analysis_results.get('typography_analysis', {}))
        validated['color'], color_errors = self._validate_color(analysis_results.get('color_analysis', {}))
        
        all_errors = copy_errors + logo_errors + typo_errors + color_errors
        
        # Determine if scoring should be blocked
        has_critical_failure = any(
            validated['copywriting'].get('critical_error', False),
            validated['logo'].get('critical_error', False),
            validated['typography'].get('critical_error', False),
            validated['color'].get('critical_error', False)
        )
        
        if has_critical_failure:
            logger.error(f"[PreScoringGuard] Critical signal failure detected - blocking overall scoring")
            validated['overall'] = {
                'status': 'blocked',
                'bucket': 'unknown',
                'compliance_score': None,
                'primary_reason': 'Critical signal failure - cannot provide compliance score',
                'critical_signal_failure': True
            }
            return False, all_errors, validated
        
        # If all validations pass, scoring can proceed
        return True, [], validated
    
    def _validate_copywriting(self, copywriting: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate copywriting analysis results.
        
        Rules:
        - OCR failure with visible text = critical error
        - Unknown/failed status = don't score
        - Critical spelling errors = compliance failure
        """
        errors = []
        validated = copywriting.copy()
        
        status = copywriting.get('status', 'passed')
        
        # RULE: OCR failure with visible text = critical error
        if status == 'failed':
            ocr_failure = copywriting.get('ocr_failure', False)
            visible_text_detected = copywriting.get('visible_text_detected', False)
            critical_error = copywriting.get('critical_error', False)
            
            if ocr_failure and visible_text_detected:
                validated['critical_error'] = True
                validated['system_error'] = True
                errors.append("CRITICAL: OCR failed despite visible text detected")
                logger.error("[PreScoringGuard] CRITICAL: OCR failure with visible text - blocking scoring")
        
        # RULE: Never score unknown or failed signals
        if status in ['failed', 'skipped', 'unknown']:
            validated['score'] = None
            validated['contributes_to_overall'] = False
            if status == 'failed':
                self.blocked_signals.append('copywriting')
        else:
            validated['contributes_to_overall'] = True
        
        # RULE: Critical spelling errors = compliance failure
        grammar_analysis = copywriting.get('grammar_analysis', {})
        if grammar_analysis.get('compliance_failed') or grammar_analysis.get('critical_errors'):
            validated['critical_error'] = True
            errors.append("CRITICAL: Copywriting compliance failed due to critical errors")
        
        return validated, errors
    
    def _validate_logo(self, logo: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate logo analysis results.
        
        Rules:
        - Logo detection failure ≠ no logo detected
        - Placement validation requires verified logo
        - Unknown/failed status = don't score
        """
        errors = []
        validated = logo.copy()
        
        status = logo.get('status', 'passed')
        detections = logo.get('logo_detections', [])
        
        # RULE: Logo detection failure ≠ no logo detected
        if status == 'failed':
            logo_errors = logo.get('errors', [])
            # Check if errors indicate detection failure vs no detections
            has_detection_error = any(
                'detection failed' in str(e).lower() or 'error' in str(e).lower()
                for e in logo_errors
            )
            
            if has_detection_error:
                validated['critical_error'] = True
                validated['detection_failure'] = True
                errors.append("CRITICAL: Logo detection failed (system error, not 'no logo')")
                logger.error("[PreScoringGuard] CRITICAL: Logo detection failure - blocking scoring")
        
        # RULE: Placement validation requires verified logo
        placement_validation = logo.get('placement_validation', {})
        placement_status = placement_validation.get('status', 'unknown')
        
        # Check if detections are verified
        verified_logos = [d for d in detections if d.get('verified', False)]
        
        if not verified_logos and placement_status not in ['not_applicable', 'unknown']:
            # Placement validation attempted but no verified logo
            validated['placement_validation'] = {
                **placement_validation,
                'status': 'not_applicable',
                'reason': 'Placement validation requires verified logo'
            }
            errors.append("Warning: Placement validation attempted but no verified logo")
        
        # RULE: Never score unknown or failed signals
        if status in ['failed', 'skipped', 'unknown']:
            validated['score'] = None
            validated['contributes_to_overall'] = False
            if status == 'failed' and validated.get('detection_failure'):
                self.blocked_signals.append('logo')
        else:
            validated['contributes_to_overall'] = True
        
        return validated, errors
    
    def _validate_typography(self, typography: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate typography analysis results.
        
        Rules:
        - Typography compliance requires brand fonts
        - Unknown status = don't score
        - Do not guess font names
        """
        errors = []
        validated = typography.copy()
        
        status = typography.get('status', 'passed')
        
        # RULE: Typography compliance requires brand fonts
        if status == 'unknown':
            expected_fonts = typography.get('expected_fonts', [])
            if not expected_fonts:
                validated['score'] = None
                validated['compliance_score'] = None
                validated['contributes_to_overall'] = False
                validated['reason'] = 'Brand font guidelines not provided - cannot validate font compliance'
                logger.warning("[PreScoringGuard] Typography status is 'unknown' - not contributing to score")
        
        # RULE: Never score unknown or failed signals
        if status in ['failed', 'skipped', 'unknown']:
            validated['score'] = None
            validated['contributes_to_overall'] = False
        else:
            # Even if passed, only observable metrics are valid (not font compliance)
            validated['contributes_to_overall'] = True
        
        # RULE: Do not guess font names
        detected_fonts = typography.get('fonts_detected', [])
        for font in detected_fonts:
            if font.get('font_family') not in ['unknown', None]:
                # If font name is guessed, mark as unknown
                font['font_family'] = 'unknown'
                font['font_identification_reliable'] = False
                errors.append(f"Warning: Font family guessed: {font.get('font_family')} - marked as unknown")
        
        return validated, errors
    
    def _validate_color(self, color: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate color analysis results.
        
        Rules:
        - Color compliance requires brand palette
        - Unknown status = don't score
        """
        errors = []
        validated = color.copy()
        
        status = color.get('status', 'passed')
        brand_palette = color.get('brand_palette', [])
        
        # RULE: Color compliance requires brand palette
        if status == 'observed_only' or not brand_palette:
            validated['score'] = None
            validated['compliance_score'] = None
            validated['contributes_to_overall'] = False
            validated['reason'] = 'Brand color palette not provided - cannot validate color compliance'
        
        # RULE: Never score unknown or failed signals
        if status in ['failed', 'skipped', 'unknown', 'observed_only']:
            validated['score'] = None
            validated['contributes_to_overall'] = False
        else:
            validated['contributes_to_overall'] = True
        
        return validated, errors
    
    def should_block_scoring(self, validated_results: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if overall scoring should be blocked.
        
        Args:
            validated_results: Validated analysis results
            
        Returns:
            Tuple: (should_block, reason)
        """
        # Check for critical errors
        critical_errors = []
        
        if validated_results.get('copywriting', {}).get('critical_error'):
            critical_errors.append("Copywriting: OCR failure or critical spelling errors")
        
        if validated_results.get('logo', {}).get('critical_error'):
            critical_errors.append("Logo: Detection failure (system error)")
        
        if validated_results.get('typography', {}).get('critical_error'):
            critical_errors.append("Typography: Critical error")
        
        if validated_results.get('color', {}).get('critical_error'):
            critical_errors.append("Color: Critical error")
        
        if critical_errors:
            reason = f"Critical signal failures: {', '.join(critical_errors)}"
            return True, reason
        
        return False, ""
    
    def get_validated_score(self, validated_results: Dict[str, Any], calculated_score: float) -> Optional[float]:
        """
        Get validated score after guard checks.
        
        Args:
            validated_results: Validated analysis results
            calculated_score: Calculated weighted average score
            
        Returns:
            Validated score (None if blocked, otherwise calculated score)
        """
        should_block, reason = self.should_block_scoring(validated_results)
        
        if should_block:
            logger.error(f"[PreScoringGuard] Scoring blocked: {reason}")
            return None
        
        return calculated_score

