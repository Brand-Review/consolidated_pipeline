"""
Deterministic Scoring Engine
Confidence-weighted penalties - no AI, pure math.

Formula:
score = 100
score -= spellingPenalty * spellingConfidence
score -= logoPenalty * logoConfidence
score -= colorPenalty * colorConfidence
score = max(0, round(score))

Penalties (max):
- Spelling: 30
- Logo placement: 40
- Color mismatch: 30
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Deterministic scoring engine with confidence-weighted penalties.
    
    Rules:
    - No AI allowed
    - Same input → same score (deterministic)
    - Confidence-weighted penalties
    - Scores capped per category
    """

    # Bucket thresholds (non-negotiable)
    APPROVE_THRESHOLD = 85
    REVIEW_THRESHOLD = 60

    # Maximum penalties (confidence-weighted)
    MAX_SPELLING_PENALTY = 30
    MAX_LOGO_PENALTY = 40
    MAX_COLOR_PENALTY = 30

    def __init__(
        self,
        approve_threshold: int = APPROVE_THRESHOLD,
        review_threshold: int = REVIEW_THRESHOLD,
    ):
        self.approve_threshold = approve_threshold
        self.review_threshold = review_threshold

    # -----------------------------
    # Text Analysis Scoring (Spelling, Grammar, Phrases)
    # -----------------------------
    def score_text_analysis(
        self,
        text_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Score text analysis (spelling, grammar, phrases) with confidence-weighted penalties.
        
        Args:
            text_result: Text analyzer result with spellingErrors, phraseIssues, grammarFlags
            
        Returns:
            Dictionary with:
            {
                "score": int,  # 0-100
                "penalty": float,  # Total penalty applied
                "spellingPenalty": float,  # Spelling penalty
                "grammarPenalty": float,  # Grammar penalty
                "phrasePenalty": float,  # Phrase penalty
                "confidence": float  # Average confidence
            }
        """
        if not text_result or text_result.get("status") != "passed":
            return {
                "score": 100,  # No penalty if analysis failed or skipped
                "penalty": 0.0,
                "spellingPenalty": 0.0,
                "grammarPenalty": 0.0,
                "phrasePenalty": 0.0,
                "confidence": 1.0
            }
        
        spelling_errors = text_result.get("spellingErrors", []) or []
        phrase_issues = text_result.get("phraseIssues", []) or []
        grammar_flags = text_result.get("grammarFlags", []) or []
        
        # Calculate penalties with confidence weighting
        spelling_penalty = self._calculate_spelling_penalty(spelling_errors)
        phrase_penalty = self._calculate_phrase_penalty(phrase_issues)
        grammar_penalty = self._calculate_grammar_penalty(grammar_flags)
        
        # Total penalty (confidence-weighted)
        total_penalty = spelling_penalty + phrase_penalty + grammar_penalty
        
        # Cap total penalty at MAX_SPELLING_PENALTY
        total_penalty = min(total_penalty, self.MAX_SPELLING_PENALTY)
        
        # Calculate score
        score = 100 - total_penalty
        score = max(0, round(score))
        
        # Calculate average confidence
        confidences = []
        for error in spelling_errors:
            confidences.append(error.get("confidence", 0.85))
        for issue in phrase_issues:
            confidences.append(issue.get("confidence", 0.85))
        for flag in grammar_flags:
            confidences.append(flag.get("confidence", 0.85))
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        
        return {
            "score": score,
            "penalty": total_penalty,
            "spellingPenalty": spelling_penalty,
            "grammarPenalty": grammar_penalty,
            "phrasePenalty": phrase_penalty,
            "confidence": avg_confidence
        }
    
    def _calculate_spelling_penalty(self, spelling_errors: List[Dict[str, Any]]) -> float:
        """Calculate spelling penalty with confidence weighting."""
        if not spelling_errors:
            return 0.0
        
        # Penalty per error: 5 points base, weighted by confidence
        total_penalty = 0.0
        for error in spelling_errors:
            confidence = error.get("confidence", 0.85)
            penalty = 5.0 * confidence  # Base 5 points, weighted by confidence
            total_penalty += penalty
        
        # Cap at reasonable max (e.g., 20 for many errors)
        return min(total_penalty, 20.0)
    
    def _calculate_phrase_penalty(self, phrase_issues: List[Dict[str, Any]]) -> float:
        """Calculate phrase error penalty with confidence weighting."""
        if not phrase_issues:
            return 0.0
        
        # Penalty per phrase error: 3 points base, weighted by confidence
        total_penalty = 0.0
        for issue in phrase_issues:
            confidence = issue.get("confidence", 0.85)
            penalty = 3.0 * confidence  # Base 3 points, weighted by confidence
            total_penalty += penalty
        
        # Cap at reasonable max (e.g., 10 for many errors)
        return min(total_penalty, 10.0)
    
    def _calculate_grammar_penalty(self, grammar_flags: List[Dict[str, Any]]) -> float:
        """Calculate grammar penalty with confidence weighting."""
        if not grammar_flags:
            return 0.0
        
        # Penalty per grammar flag: 2 points base, weighted by confidence
        total_penalty = 0.0
        for flag in grammar_flags:
            confidence = flag.get("confidence", 0.85)
            penalty = 2.0 * confidence  # Base 2 points, weighted by confidence
            total_penalty += penalty
        
        # Cap at reasonable max (e.g., 10 for many flags)
        return min(total_penalty, 10.0)

    # Legacy methods removed - use new signal-based methods instead
    # score_grammar, score_tone, score_brand are replaced by:
    # - score_text_analysis (spelling, grammar, phrases)
    # - score_logo_placement (logo violations)
    # - score_colors (color compliance)

    # -----------------------------
    # Logo Placement Scoring
    # -----------------------------
    def score_logo_placement(
        self,
        logo_violations: Optional[Dict[str, Any]] = None,
        logo_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Score logo placement with confidence-weighted penalties.
        
        Args:
            logo_violations: Violations result from LogoPlacementViolations.detect_violations()
            logo_detected: Whether logo was detected
            
        Returns:
            Dictionary with:
            {
                "score": int,  # 0-100
                "penalty": float,  # Total penalty applied
                "placementPenalty": float,  # Placement violation penalty
                "sizePenalty": float,  # Size violation penalty
                "edgePenalty": float,  # Edge padding penalty
                "confidence": float  # Average confidence
            }
        """
        if not logo_detected:
            # Logo not detected - maximum penalty
            return {
                "score": 60,  # Base score when logo missing (100 - 40)
                "penalty": 40.0,
                "placementPenalty": 40.0,
                "sizePenalty": 0.0,
                "edgePenalty": 0.0,
                "confidence": 1.0
            }
        
        if not logo_violations:
            # Logo detected but no violations - perfect score
            return {
                "score": 100,
                "penalty": 0.0,
                "placementPenalty": 0.0,
                "sizePenalty": 0.0,
                "edgePenalty": 0.0,
                "confidence": 1.0
            }
        
        # Calculate penalties with confidence weighting
        placement_penalty = 0.0
        size_penalty = 0.0
        edge_penalty = 0.0
        
        violations = logo_violations.get("violations", [])
        zone_info = logo_violations.get("zoneInfo", {})
        confidence = zone_info.get("confidence", 0.85)
        
        # Placement violations
        if not logo_violations.get("placementOk", True):
            placement_penalty = 15.0 * confidence  # Base 15 points, weighted by confidence
        
        # Size violations
        if not logo_violations.get("sizeOk", True):
            size_penalty = 15.0 * confidence  # Base 15 points, weighted by confidence
        
        # Edge padding violations
        if not logo_violations.get("edgePaddingOk", True):
            edge_penalty = 10.0 * confidence  # Base 10 points, weighted by confidence
        
        # Total penalty (capped at MAX_LOGO_PENALTY)
        total_penalty = placement_penalty + size_penalty + edge_penalty
        total_penalty = min(total_penalty, self.MAX_LOGO_PENALTY)
        
        # Calculate score
        score = 100 - total_penalty
        score = max(0, round(score))
        
        return {
            "score": score,
            "penalty": total_penalty,
            "placementPenalty": placement_penalty,
            "sizePenalty": size_penalty,
            "edgePenalty": edge_penalty,
            "confidence": confidence
        }
    
    # -----------------------------
    # Color Compliance Scoring
    # -----------------------------
    def score_colors(
        self,
        vision_result: Optional[Dict[str, Any]] = None,
        brand_colors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Score color compliance with confidence-weighted penalties.
        
        Args:
            vision_result: Vision analyzer result with colors
            brand_colors: List of brand-approved color hex codes (optional)
            
        Returns:
            Dictionary with:
            {
                "score": int,  # 0-100
                "penalty": float,  # Total penalty applied
                "confidence": float  # Average confidence
            }
        """
        if not vision_result or vision_result.get("status") != "passed":
            return {
                "score": 100,  # No penalty if analysis failed or skipped
                "penalty": 0.0,
                "confidence": 1.0
            }
        
        colors = vision_result.get("colors", {})
        dominant_colors = colors.get("dominant", [])
        
        if not dominant_colors:
            return {
                "score": 100,  # No colors detected - no penalty
                "penalty": 0.0,
                "confidence": 1.0
            }
        
        # If brand colors are provided, check for matches
        if brand_colors:
            # Check if any dominant colors match brand colors
            matched = False
            for color in dominant_colors:
                color_hex = color.get("hex", "").upper()
                if color_hex in [c.upper() for c in brand_colors]:
                    matched = True
                    break
            
            if not matched:
                # Color mismatch - apply penalty
                color_confidence = dominant_colors[0].get("confidence", 0.85) if dominant_colors else 0.85
                penalty = self.MAX_COLOR_PENALTY * color_confidence
                score = 100 - penalty
                score = max(0, round(score))
                
                return {
                    "score": score,
                    "penalty": penalty,
                    "confidence": color_confidence
                }
        
        # No brand colors specified or colors match - perfect score
        return {
            "score": 100,
            "penalty": 0.0,
            "confidence": 1.0
        }

    # -----------------------------
    # Overall Score Calculation (Deterministic)
    # -----------------------------
    def calculate_overall_score(
        self,
        text_score: Dict[str, Any],
        logo_score: Dict[str, Any],
        color_score: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall compliance score using confidence-weighted penalties.
        
        Formula:
        score = 100
        score -= spellingPenalty * spellingConfidence
        score -= logoPenalty * logoConfidence
        score -= colorPenalty * colorConfidence
        score = max(0, round(score))
        
        Args:
            text_score: Text analysis score result
            logo_score: Logo placement score result
            color_score: Color compliance score result
            
        Returns:
            Dictionary with overall score and bucket:
            {
                "overallCompliance": int,  # 0-100
                "bucket": str,  # "approve" | "review" | "reject"
                "textScore": int,
                "logoScore": int,
                "colorScore": int,
                "penalties": {
                    "text": float,
                    "logo": float,
                    "color": float
                }
            }
        """
        # Extract scores
        text_score_val = text_score.get("score", 100)
        logo_score_val = logo_score.get("score", 100)
        color_score_val = color_score.get("score", 100)
        
        # Calculate weighted average (or use formula-based approach)
        # Since we already applied penalties, we can use weighted average
        # Or use the formula: score = 100 - (sum of weighted penalties)
        
        # Extract penalties (already confidence-weighted)
        text_penalty = text_score.get("penalty", 0.0)
        logo_penalty = logo_score.get("penalty", 0.0)
        color_penalty = color_score.get("penalty", 0.0)
        
        # Calculate overall score using formula
        base_score = 100
        total_penalty = text_penalty + logo_penalty + color_penalty
        
        # Cap total penalty (shouldn't exceed 100)
        total_penalty = min(total_penalty, 100.0)
        
        overall_score = base_score - total_penalty
        overall_score = max(0, round(overall_score))
        
        # Determine bucket
        bucket = self.get_bucket(overall_score)
        
        return {
            "overallCompliance": overall_score,
            "bucket": bucket,
            "textScore": text_score_val,
            "logoScore": logo_score_val,
            "colorScore": color_score_val,
            "penalties": {
                "text": text_penalty,
                "logo": logo_penalty,
                "color": color_penalty
            }
        }
    
    def get_bucket(self, score: Optional[int]) -> str:
        """
        Bucket decision based on fixed thresholds.
        
        Args:
            score: Overall compliance score (0-100) or None for unknown
            
        Returns:
            Bucket name: "approve" | "review" | "reject" | "unknown"
        """
        if score is None:
            return "unknown"
        
        if score >= self.approve_threshold:
            return "approve"
        if score >= self.review_threshold:
            return "review"
        return "reject"

    # -----------------------------
    # Explanation builder
    # -----------------------------
    def explain(self, scores: Dict[str, Any], normalized: Optional[Dict[str, Any]] = None) -> list:
        """
        Build human-readable reasons based on scores.
        Plain English only - no AI language.
        """
        reasons = []
        normalized = normalized or {}
        
        grammar_score = scores.get("grammar", 100)
        brand_score = scores.get("brand", 100)
        tone_score = scores.get("tone", 100)
        
        # Grammar explanations (specific)
        if grammar_score < 70:
            grammar = normalized.get("grammar", {}) if normalized else {}
            error_count = len(grammar.get("errors", []))
            spelling_count = len(grammar.get("spelling", []))
            punctuation_count = len(grammar.get("punctuation", []))
            
            if error_count > 0:
                reasons.append(f"{error_count} grammar error{'s' if error_count > 1 else ''} detected")
            if spelling_count > 0:
                reasons.append(f"{spelling_count} spelling mistake{'s' if spelling_count > 1 else ''} found")
            if punctuation_count > 0:
                reasons.append(f"{punctuation_count} punctuation issue{'s' if punctuation_count > 1 else ''} found")
            if not any([error_count, spelling_count, punctuation_count]) and grammar_score < 70:
                reasons.append("Grammar issues detected")
        
        # Brand explanations (specific)
        if brand_score < 70:
            brand = normalized.get("brand", {}) if normalized else {}
            logo = brand.get("logo", {})
            colors = brand.get("colors", {})
            
            if not logo.get("present", True):
                reasons.append("Logo missing")
            elif not logo.get("size_ok", True):
                reasons.append("Logo size incorrect")
            elif not logo.get("placement_ok", True):
                reasons.append("Logo placement incorrect")
            
            if not colors.get("primary_match", True):
                reasons.append("Primary color mismatch")
            
            # If no specific issues found but score is low, provide general message
            if not any([
                not logo.get("present", True),
                not logo.get("size_ok", True),
                not logo.get("placement_ok", True),
                not colors.get("primary_match", True)
            ]):
                reasons.append("Brand guideline violations")
        
        # Tone explanations (specific)
        if tone_score < 60:
            tone = normalized.get("tone", {}) if normalized else {}
            detected_tone = tone.get("tone", "unknown")
            if detected_tone == "unknown":
                reasons.append("Tone could not be determined")
            else:
                reasons.append("Tone misalignment detected")
        
        # If no specific reasons, provide general feedback
        if not reasons:
            if scores.get("bucket") == "approve":
                reasons.append("Content meets brand guidelines")
            else:
                reasons.append("Review recommended")
        
        return reasons

    # -----------------------------
    # New Signal-Based Scoring API
    # -----------------------------
    def score_signals(
        self,
        text_result: Optional[Dict[str, Any]] = None,
        vision_result: Optional[Dict[str, Any]] = None,
        logo_violations: Optional[Dict[str, Any]] = None,
        logo_detected: bool = False,
        brand_colors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Score signals from analyzers (new signal-based API).
        
        Args:
            text_result: Text analyzer result (spellingErrors, phraseIssues, grammarFlags)
            vision_result: Vision analyzer result (colors)
            logo_violations: Logo placement violations result
            logo_detected: Whether logo was detected
            brand_colors: List of brand-approved color hex codes (optional)
            
        Returns:
            Dictionary with overall score, bucket, and component scores
        """
        # Score each component
        text_score = self.score_text_analysis(text_result)
        logo_score = self.score_logo_placement(logo_violations, logo_detected)
        color_score = self.score_colors(vision_result, brand_colors)
        
        # Calculate overall score
        overall_result = self.calculate_overall_score(text_score, logo_score, color_score)
        
        return {
            "overallCompliance": overall_result["overallCompliance"],
            "bucket": overall_result["bucket"],
            "textScore": text_score,
            "logoScore": logo_score,
            "colorScore": color_score,
            "penalties": overall_result["penalties"],
            "thresholds": {
                "approve": self.approve_threshold,
                "review": self.review_threshold
            }
        }