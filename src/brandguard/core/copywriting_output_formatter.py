"""
Copywriting Output Formatter

Refactored to produce human-readable, actionable output that:
- Hides technical details (OCR confidence, probabilities, raw analyzer output)
- Filters false positives (numbers, product names, currencies, UI labels)
- Groups errors by sentence/headline
- Surfaces OCR noise as warnings
- Explains WHY changes matter in business terms
- Follows strict JSON schema
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# UI labels and domain terms that should NEVER be flagged as errors
UI_TERMS = {
    "dashboard", "invoice", "usd", "vat", "tax", "crm", "api", "ui", "ux",
    "subtotal", "total", "qty", "id", "order", "web", "login", "signup",
    "signin", "logout", "signout", "checkout", "cart", "price", "cost",
    "fee", "paid", "due", "status", "pending", "completed", "cancelled",
    "email", "phone", "address", "name", "date", "time", "hour", "minute",
    "second", "day", "week", "month", "year", "today", "yesterday", "tomorrow"
}

# Currency codes
CURRENCY_CODES = {"usd", "eur", "gbp", "jpy", "cad", "aud", "chf", "cny", "inr"}

# Common product name patterns (to be extended)
PRODUCT_PATTERNS = [
    r'^[A-Z]{2,}\d+',  # Acronyms with numbers (e.g., "CRM2024")
    r'^\d+[A-Z]+',     # Numbers with letters (e.g., "2024CRM")
]


class CopywritingOutputFormatter:
    """Formats copywriting analysis output for human consumption"""
    
    @staticmethod
    def format_output(
        raw_analysis: Dict[str, Any],
        text_content: str,
        ocr_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format copywriting analysis output following strict schema.
        
        Args:
            raw_analysis: Raw analyzer output
            text_content: Extracted text content
            ocr_result: OCR result (for detecting noise and confidence)
            
        Returns:
            Formatted output matching target schema
        """
        # Determine text source
        text_source = CopywritingOutputFormatter._determine_text_source(raw_analysis, ocr_result)
        
        # Extract OCR reliability score (0-1)
        ocr_reliability = CopywritingOutputFormatter._extract_ocr_reliability(ocr_result)
        
        # Get all spelling errors (from multiple sources)
        all_spelling_errors = CopywritingOutputFormatter._collect_all_spelling_errors(raw_analysis)
        
        # Detect OCR noise
        ocr_warnings = CopywritingOutputFormatter._detect_ocr_noise(text_content, ocr_result)
        
        # Filter and group spelling errors
        grammar_errors = CopywritingOutputFormatter._format_grammar_errors(
            raw_analysis, text_content
        )
        
        # Determine grammar analysis status
        grammar_status = CopywritingOutputFormatter._determine_grammar_status(
            text_source, all_spelling_errors, grammar_errors, ocr_reliability
        )
        
        # Generate grammar summary (one of 3 specific messages)
        grammar_summary = CopywritingOutputFormatter._generate_grammar_summary(
            grammar_errors, all_spelling_errors, text_source, ocr_reliability
        )
        
        # Build final copy verdict
        final_copy_verdict = CopywritingOutputFormatter._build_final_copy_verdict(
            all_spelling_errors, grammar_errors, text_source, ocr_reliability
        )
        
        # Filter out false positives from errors list
        filtered_errors = CopywritingOutputFormatter._filter_false_positives(
            raw_analysis.get('errors', [])
        )
        
        # Build compliance object
        compliance = CopywritingOutputFormatter._build_compliance_object(
            raw_analysis, grammar_errors, all_spelling_errors, text_source, ocr_reliability
        )
        
        # Build tone analysis
        tone_analysis = CopywritingOutputFormatter._build_tone_analysis(
            raw_analysis.get('tone_analysis', {})
        )
        
        # Build text metrics
        text_metrics = CopywritingOutputFormatter._build_text_metrics(text_content)
        
        # Generate recommendations
        recommendations = CopywritingOutputFormatter._generate_recommendations(
            grammar_errors, ocr_warnings, compliance, ocr_reliability, text_source
        )
        
        # Calculate copywriting score (0-100)
        copywriting_score = CopywritingOutputFormatter._calculate_score(
            raw_analysis.get('copywriting_score', 0.0),
            raw_analysis.get('copywritingScore', 0.0)
        )
        
        has_issues = len(grammar_errors) > 0 or len(all_spelling_errors) > 0
        return {
            'status': 'issues_found' if has_issues else 'clean',
            'issues': grammar_errors,
            'compliance': compliance,
            'copywritingScore': copywriting_score,
            'errors': filtered_errors,  # Only real errors, no false positives
            'extractedText': text_content,
            'grammarAnalysis': {
                'errors': grammar_errors,
                'grammarScore': compliance.get('score', 100),
                'status': grammar_status,  # NEW: "limited" | "skipped" | "completed"
                'summary': grammar_summary  # One of 3 specific messages
            },
            'recommendations': recommendations,
            'textContent': text_content,
            'textMetrics': text_metrics,
            'toneAnalysis': tone_analysis,
            'warnings': ocr_warnings,  # OCR noise as warnings, not errors
            # NEW FIELDS
            'textSource': text_source,  # "image" | "document" | "html"
            'ocrReliability': ocr_reliability,  # 0-1 score
            'finalCopyVerdict': final_copy_verdict  # Final authority object
        }
    
    @staticmethod
    def _detect_ocr_noise(text: str, ocr_result: Optional[Dict[str, Any]]) -> List[str]:
        """
        Detect OCR noise and return as warnings.
        
        OCR noise indicators:
        - Random characters mixed with text
        - Repeated characters (e.g., "aaaaa")
        - Unusual character sequences
        - Low confidence words (if available)
        """
        warnings = []
        
        if not text:
            return warnings
        
        # Check for repeated characters (likely OCR artifact)
        if re.search(r'(.)\1{4,}', text):
            warnings.append("OCR may have introduced character repetition. Review extracted text for accuracy.")
        
        # Check for unusual character sequences
        if re.search(r'[^\w\s\.\,\!\?\-]{3,}', text):
            warnings.append("Unusual character sequences detected. Some text may be OCR noise.")
        
        # Check OCR confidence if available
        if ocr_result:
            ocr_confidence = ocr_result.get('confidence', 1.0)
            if ocr_confidence < 0.7:
                warnings.append(f"OCR confidence is low ({ocr_confidence:.0%}). Some text may be inaccurate.")
        
        return warnings
    
    @staticmethod
    def _format_grammar_errors(
        raw_analysis: Dict[str, Any],
        text_content: str
    ) -> List[Dict[str, Any]]:
        """
        Format grammar errors grouped by sentence/headline.
        
        Returns:
            List of error objects with:
            - errorType: "spelling" | "grammar" | "style"
            - explanation: Business explanation of WHY it matters
            - originalText: The problematic sentence/headline (not just word)
            - severity: "critical" | "high" | "medium" | "low"
            - suggestedFix: The correction
        """
        errors = []
        
        # Get spelling errors from various locations
        spelling_errors = raw_analysis.get('spellingErrors', [])
        grammar_analysis = raw_analysis.get('grammar_analysis', {})
        spelling_violations = grammar_analysis.get('spellingViolations', [])
        
        # Combine all spelling errors
        all_spelling_errors = spelling_errors + spelling_violations
        
        # Get text structure for grouping
        text_structure = raw_analysis.get('text_structure', {})
        if not text_structure:
            # Try to identify structure if not provided
            try:
                from ..utils.text_structure_analyzer import identify_text_structure
                text_structure = identify_text_structure(text_content)
            except:
                text_structure = {}
        
        # Group errors by component (headline, subtext, cta, body)
        errors_by_component = defaultdict(list)
        
        for error in all_spelling_errors:
            if not isinstance(error, dict):
                continue
            
            # Filter false positives
            word = error.get('word', '')
            if CopywritingOutputFormatter._is_false_positive(word):
                continue
            
            # Determine location/component
            location = error.get('location', 'body')
            component = text_structure.get(location, '')
            
            # If component not found, try to find which component contains the word
            if not component:
                try:
                    from ..utils.text_structure_analyzer import get_text_component_for_word
                    component_name = get_text_component_for_word(word, text_structure)
                    if component_name:
                        component = text_structure.get(component_name, '')
                        location = component_name
                except:
                    pass
            
            # Determine severity
            severity = error.get('severity', 'medium')
            if isinstance(severity, str):
                severity = severity.lower()
                if severity == 'critical':
                    severity = 'critical'
                elif severity == 'high':
                    severity = 'high'
                elif severity == 'medium':
                    severity = 'medium'
                else:
                    severity = 'low'
            else:
                severity = 'medium'
            
            # Group by component
            errors_by_component[location].append({
                'word': word,
                'correction': error.get('correction') or error.get('suggestion', ''),
                'severity': severity
            })
        
        # Build error objects grouped by component
        for location, component_errors in errors_by_component.items():
            # Get the full text of this component
            component_text = text_structure.get(location, '')
            if not component_text:
                # Fallback: use first error word's sentence
                component_text = component_errors[0]['word']
            
            # Determine overall severity (use highest)
            severities = [e['severity'] for e in component_errors]
            if 'critical' in severities:
                overall_severity = 'critical'
            elif 'high' in severities:
                overall_severity = 'high'
            elif 'medium' in severities:
                overall_severity = 'medium'
            else:
                overall_severity = 'low'
            
            # Build combined error message
            error_words = [e['word'] for e in component_errors]
            corrections = [e['correction'] for e in component_errors if e['correction']]
            
            # Create one error object per component (grouped)
            if len(component_errors) == 1:
                # Single error
                error_obj = {
                    'errorType': 'spelling',
                    'explanation': CopywritingOutputFormatter._generate_business_explanation(
                        component_errors[0]['word'],
                        component_errors[0]['correction'],
                        location,
                        overall_severity
                    ),
                    'originalText': component_text,
                    'severity': overall_severity,
                    'suggestedFix': component_errors[0]['correction'] or ''
                }
            else:
                # Multiple errors in same component - group them
                fixes = [f"{e['word']} → {e['correction']}" for e in component_errors if e['correction']]
                error_obj = {
                    'errorType': 'spelling',
                    'explanation': CopywritingOutputFormatter._generate_business_explanation(
                        f"{len(component_errors)} spelling errors",
                        ', '.join(fixes) if fixes else 'multiple errors',
                        location,
                        overall_severity
                    ),
                    'originalText': component_text,
                    'severity': overall_severity,
                    'suggestedFix': '; '.join(fixes) if fixes else 'Review text for spelling errors'
                }
            
            errors.append(error_obj)
        
        # Sort by severity (critical first)
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        errors.sort(key=lambda e: severity_order.get(e.get('severity', 'low'), 3))
        
        return errors
    
    @staticmethod
    def _is_false_positive(word: str) -> bool:
        """
        Check if a word is a false positive (number, product name, currency, UI label).
        """
        if not word:
            return True
        
        word_lower = word.lower()
        
        # Check UI terms
        if word_lower in UI_TERMS:
            return True
        
        # Check currency codes
        if word_lower in CURRENCY_CODES:
            return True
        
        # Check if it's a number
        if word.replace('.', '').replace(',', '').isdigit():
            return True
        
        # Check product name patterns
        for pattern in PRODUCT_PATTERNS:
            if re.match(pattern, word):
                return True
        
        # Check if it's all uppercase (likely acronym)
        if word.isupper() and len(word) >= 2:
            return True
        
        return False
    
    @staticmethod
    def _generate_business_explanation(
        word: str,
        correction: Optional[str],
        location: str,
        severity: str
    ) -> str:
        """
        Generate business explanation of WHY the error matters.
        """
        location_map = {
            'headline': 'headline',
            'cta': 'call-to-action button',
            'subtext': 'subheading',
            'body': 'body text'
        }
        location_name = location_map.get(location, 'text')
        
        if severity == 'critical':
            if correction:
                return f"Spelling error '{word}' in {location_name} should be '{correction}'. Headline errors damage brand credibility and can cause customer confusion, directly impacting conversion rates."
            else:
                return f"Unknown word '{word}' in {location_name}. Headline errors damage brand credibility and can cause customer confusion, directly impacting conversion rates."
        
        elif severity == 'high':
            if correction:
                return f"Spelling error '{word}' in {location_name} should be '{correction}'. CTA errors reduce user trust and can prevent customers from taking action, lowering conversion rates."
            else:
                return f"Unknown word '{word}' in {location_name}. CTA errors reduce user trust and can prevent customers from taking action, lowering conversion rates."
        
        elif severity == 'medium':
            if correction:
                return f"Spelling error '{word}' in {location_name} should be '{correction}'. Subheading errors can reduce message clarity and impact user engagement."
            else:
                return f"Unknown word '{word}' in {location_name}. Subheading errors can reduce message clarity and impact user engagement."
        
        else:  # low
            if correction:
                return f"Spelling error '{word}' in {location_name} should be '{correction}'. While less critical, body text errors can still impact perceived brand quality."
            else:
                return f"Unknown word '{word}' in {location_name}. While less critical, body text errors can still impact perceived brand quality."
    
    @staticmethod
    def _filter_false_positives(errors: List[Any]) -> List[str]:
        """
        Filter out false positives from errors list.
        Only return human-readable, actionable errors.
        """
        filtered = []
        
        for error in errors:
            if isinstance(error, str):
                # Check if error mentions false positives
                error_lower = error.lower()
                if any(term in error_lower for term in ['ocr', 'confidence', 'probability', 'model']):
                    continue  # Skip technical errors
                filtered.append(error)
            elif isinstance(error, dict):
                # Extract meaningful error message
                word = error.get('word', '')
                if word and not CopywritingOutputFormatter._is_false_positive(word):
                    correction = error.get('correction') or error.get('suggestion', '')
                    if correction:
                        filtered.append(f"{word} → {correction}")
        
        return filtered
    
    @staticmethod
    def _build_compliance_object(
        raw_analysis: Dict[str, Any],
        grammar_errors: List[Dict[str, Any]],
        spelling_errors: List[Any],
        text_source: str,
        ocr_reliability: float
    ) -> Dict[str, Any]:
        """
        Build compliance object with explanations, failures, and score.
        
        CRITICAL: Block "clean" conclusions when textSource === "image" and spellingErrors.length > 0
        """
        brand_voice = raw_analysis.get('brand_voice_compliance', {})
        compliance_score = brand_voice.get('score', 100.0)
        
        # Convert 0-1 to 0-100 if needed
        if compliance_score <= 1.0:
            compliance_score = compliance_score * 100.0
        
        # Extract failures
        failures = brand_voice.get('failures', [])
        if not failures:
            failures = []
        
        # Generate explanations
        explanations = []
        if grammar_errors:
            critical_errors = [e for e in grammar_errors if e.get('severity') == 'critical']
            if critical_errors:
                explanations.append(f"{len(critical_errors)} critical spelling error(s) in headline. Headline errors damage brand credibility and impact conversion rates.")
        
        # Generate failure summary
        # CRITICAL: Block "clean" conclusions when textSource === "image" and spellingErrors.length > 0
        if failures:
            failure_summary = f"{len(failures)} compliance issue(s) found. Review errors and recommendations."
        elif grammar_errors or len(spelling_errors) > 0:
            total_issues = len(grammar_errors) + len(spelling_errors)
            if text_source == "image" and ocr_reliability < 0.7:
                failure_summary = f"{total_issues} potential grammar/spelling issue(s) detected. Analysis may be affected by OCR quality."
            else:
                failure_summary = f"{total_issues} grammar/spelling issue(s) found. Review errors for details."
        else:
            # Only say "No issues found" if text source is document/HTML
            if text_source == "document" or text_source == "html":
                failure_summary = "No compliance issues found."
            else:
                failure_summary = "No compliance issues detected. Review text manually for accuracy."
        
        return {
            'explanations': explanations,
            'failureSummary': failure_summary,
            'failures': failures,
            'score': round(compliance_score, 1)
        }
    
    @staticmethod
    def _build_tone_analysis(tone_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build tone analysis object, removing technical details.
        """
        formality = tone_data.get('formality', {})
        readability = tone_data.get('readability', {})
        sentiment = tone_data.get('sentiment', {})
        
        return {
            'formality': {
                'formalityLevel': formality.get('formality_level', 'neutral'),
                'formalityScore': formality.get('formality_score', 0.5)
            },
            'readability': {
                'level': readability.get('level', 'medium'),
                'score': readability.get('score', 0.5)
            },
            'sentiment': {
                'compound': sentiment.get('compound', 0.0),
                'overallSentiment': sentiment.get('overall', 'neutral')
            }
        }
    
    @staticmethod
    def _build_text_metrics(text_content: str) -> Dict[str, Any]:
        """
        Build text metrics object.
        """
        if not text_content:
            return {
                'readabilityLevel': 'unknown',
                'sentenceCount': 0,
                'wordCount': 0
            }
        
        sentences = re.split(r'[.!?]+', text_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text_content.split()
        
        # Determine readability level (simplified)
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        if avg_words_per_sentence < 10:
            readability_level = 'easy'
        elif avg_words_per_sentence < 20:
            readability_level = 'medium'
        else:
            readability_level = 'difficult'
        
        return {
            'readabilityLevel': readability_level,
            'sentenceCount': len(sentences),
            'wordCount': len(words)
        }
    
    @staticmethod
    def _generate_recommendations(
        grammar_errors: List[Dict[str, Any]],
        ocr_warnings: List[str],
        compliance: Dict[str, Any],
        ocr_reliability: float,
        text_source: str
    ) -> List[str]:
        """
        Generate actionable recommendations.
        """
        recommendations = []
        
        # Add recommendations based on errors
        critical_errors = [e for e in grammar_errors if e.get('severity') == 'critical']
        if critical_errors:
            recommendations.append(f"Fix {len(critical_errors)} critical spelling error(s) in headline to maintain brand credibility.")
        
        high_errors = [e for e in grammar_errors if e.get('severity') == 'high']
        if high_errors:
            recommendations.append(f"Fix {len(high_errors)} spelling error(s) in call-to-action to improve conversion rates.")
        
        # Add OCR warnings as recommendations
        if ocr_warnings:
            recommendations.append("Review extracted text for OCR accuracy. Some characters may be misread.")
        
        # Add reliability-based recommendations
        if text_source == "image" and ocr_reliability < 0.7:
            recommendations.append("OCR reliability is low. Manually review extracted text for accuracy before making changes.")
        
        # Add compliance recommendations
        if compliance.get('score', 100) < 80:
            recommendations.append("Improve brand voice compliance to better align with brand guidelines.")
        
        if not recommendations:
            if text_source == "document" or text_source == "html":
                recommendations.append("Text quality is good. Continue maintaining high standards.")
            else:
                recommendations.append("Review text manually for accuracy. OCR quality may affect analysis.")
        
        return recommendations
    
    @staticmethod
    def _generate_grammar_summary(
        grammar_errors: List[Dict[str, Any]],
        spelling_errors: List[Any],
        text_source: str,
        ocr_reliability: float
    ) -> str:
        """
        Generate grammar summary - MUST be one of 3 specific messages.
        
        CRITICAL: NEVER say "No grammar or spelling errors found" when spellingErrors exist.
        
        Returns one of:
        - "Grammar analysis limited due to OCR noise."
        - "Grammar analysis skipped (image-based text)."
        - "Grammar analysis completed on clean text."
        """
        # CRITICAL RULE: Never say "No errors found" when spellingErrors exist
        if len(spelling_errors) > 0:
            # Spelling errors exist - grammar analysis is limited
            return "Grammar analysis limited due to OCR noise."
        
        # Check if analysis was skipped
        if text_source == "image" and ocr_reliability < 0.5:
            return "Grammar analysis skipped (image-based text)."
        
        # Check if we have grammar errors
        if len(grammar_errors) > 0:
            # Has errors but no spelling errors - still completed
            return "Grammar analysis completed on clean text."
        
        # No errors and clean text
        if text_source == "document" or text_source == "html":
            return "Grammar analysis completed on clean text."
        
        # Image with no errors but good OCR
        if text_source == "image" and ocr_reliability >= 0.7:
            return "Grammar analysis completed on clean text."
        
        # Default: limited due to OCR uncertainty
        return "Grammar analysis limited due to OCR noise."
    
    @staticmethod
    def _calculate_score(score1: float, score2: float) -> float:
        """
        Calculate copywriting score, converting to 0-100 scale.
        """
        score = score1 or score2 or 0.0
        
        # Convert 0-1 to 0-100 if needed
        if score <= 1.0:
            score = score * 100.0
        
        return round(max(0.0, min(100.0, score)), 1)
    
    @staticmethod
    def _determine_text_source(
        raw_analysis: Dict[str, Any],
        ocr_result: Optional[Dict[str, Any]]
    ) -> str:
        """
        Determine text source: "image" | "document" | "html"
        """
        # Check if OCR was used (indicates image source)
        if ocr_result:
            return "image"
        
        # Check raw analysis for source hints
        source = raw_analysis.get('source', '').lower()
        if 'ocr' in source or 'image' in source:
            return "image"
        elif 'document' in source or 'pdf' in source:
            return "document"
        elif 'html' in source or 'web' in source:
            return "html"
        
        # Default: assume image if OCR result exists, otherwise document
        return "image" if ocr_result else "document"
    
    @staticmethod
    def _extract_ocr_reliability(ocr_result: Optional[Dict[str, Any]]) -> float:
        """
        Extract OCR reliability score (0-1).
        
        Returns:
            0-1 score where:
            - 1.0 = perfect reliability (document/HTML, no OCR)
            - 0.8-0.9 = high reliability (good OCR)
            - 0.6-0.7 = medium reliability (acceptable OCR)
            - < 0.6 = low reliability (noisy OCR)
        """
        if not ocr_result:
            return 1.0  # Default to perfect reliability if no OCR result (document/HTML)
        
        # Try multiple possible keys
        confidence = ocr_result.get('confidence')
        if confidence is not None:
            # Convert to 0-1 if needed
            if confidence > 1.0:
                confidence = confidence / 100.0
            return round(max(0.0, min(1.0, float(confidence))), 2)
        
        # Try average confidence from words
        words = ocr_result.get('words', [])
        if words:
            confidences = []
            for word in words:
                if isinstance(word, dict):
                    word_conf = word.get('confidence')
                    if word_conf is not None:
                        if word_conf > 1.0:
                            word_conf = word_conf / 100.0
                        confidences.append(float(word_conf))
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                return round(max(0.0, min(1.0, avg_confidence)), 2)
        
        # Default reliability for image-based OCR (medium)
        return 0.7  # Medium reliability if no explicit confidence available
    
    @staticmethod
    def _determine_grammar_status(
        text_source: str,
        spelling_errors: List[Any],
        grammar_errors: List[Dict[str, Any]],
        ocr_reliability: float
    ) -> str:
        """
        Determine grammar analysis status.
        
        Rules:
        - "limited" → IF textSource === "image" AND spellingErrors.length > 0
        - "skipped" → IF textSource === "image" AND ocrReliability < 0.5
        - "completed" → Otherwise (clean text or document/HTML)
        """
        # CRITICAL RULE: IF textSource === "image" AND spellingErrors.length > 0 THEN status = "limited"
        if text_source == "image" and len(spelling_errors) > 0:
            return "limited"
        
        # Skip if OCR reliability is very low
        if text_source == "image" and ocr_reliability < 0.5:
            return "skipped"
        
        # Otherwise completed
        return "completed"
    
    @staticmethod
    def _collect_all_spelling_errors(raw_analysis: Dict[str, Any]) -> List[Any]:
        """
        Collect all spelling errors from multiple sources.
        """
        errors = []
        
        # From spellingErrors
        spelling_errors = raw_analysis.get('spellingErrors', [])
        if spelling_errors:
            errors.extend(spelling_errors)
        
        # From grammar_analysis.spellingViolations
        grammar_analysis = raw_analysis.get('grammar_analysis', {})
        spelling_violations = grammar_analysis.get('spellingViolations', [])
        if spelling_violations:
            errors.extend(spelling_violations)
        
        # From grammarErrors (if they contain spelling errors)
        grammar_errors = grammar_analysis.get('grammarErrors', [])
        for error in grammar_errors:
            if isinstance(error, dict):
                error_type = error.get('type', '').lower()
                if 'spelling' in error_type or 'misspelling' in error_type:
                    errors.append(error)
        
        return errors
    
    @staticmethod
    def _build_final_copy_verdict(
        spelling_errors: List[Any],
        grammar_errors: List[Dict[str, Any]],
        text_source: str,
        ocr_reliability: float
    ) -> Dict[str, Any]:
        """
        Build final copy verdict - the authoritative object.
        
        Rules:
        - If spellingErrors exist → status = "needs_revision"
        - If only grammar errors exist → status = "issues_found"
        - Confidence must be < 1.0 for image-based text
        - Otherwise status = "clean"
        """
        # CRITICAL RULE: If spellingErrors exist → status = "needs_revision"
        has_spelling_errors = len(spelling_errors) > 0
        has_grammar_errors = len(grammar_errors) > 0
        
        if has_spelling_errors:
            status = "needs_revision"
        elif has_grammar_errors:
            status = "issues_found"
        else:
            status = "clean"
        
        # CRITICAL RULE: Confidence must be < 1.0 for image-based text
        if text_source == "image":
            confidence = min(0.99, ocr_reliability)  # Always < 1.0 for images
        else:
            confidence = 1.0  # Perfect confidence for documents/HTML
        
        # Collect all issues
        detected_issues = []
        for error in spelling_errors:
            if isinstance(error, dict):
                detected_issues.append({
                    'type': 'spelling',
                    'word': error.get('word', ''),
                    'correction': error.get('correction') or error.get('suggestion', ''),
                    'severity': error.get('severity', 'medium'),
                    'location': error.get('location', 'body')
                })
            else:
                detected_issues.append({'type': 'spelling', 'word': str(error)})
        
        for error in grammar_errors:
            if isinstance(error, dict):
                detected_issues.append({
                    'type': error.get('errorType', 'grammar'),
                    'text': error.get('originalText', ''),
                    'explanation': error.get('explanation', ''),
                    'severity': error.get('severity', 'medium'),
                    'suggestedFix': error.get('suggestedFix', '')
                })
        
        return {
            'status': status,  # "issues_detected" | "clean"
            'confidence': round(confidence, 2),  # < 1.0 for images, 1.0 for documents
            'textSource': text_source,
            'ocrReliability': round(ocr_reliability, 2),
            'detectedIssues': detected_issues,
            'issueCount': len(detected_issues)
        }

