"""
Copywriting Analysis Module
Handles tone analysis and brand voice validation
"""

import logging
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import re

logger = logging.getLogger(__name__)

class CopywritingAnalyzer:
    """Handles copywriting analysis including tone analysis and brand voice validation"""
    
    def __init__(self, settings, imported_models: Dict[str, Any]):
        """Initialize the copywriting analyzer"""
        self.settings = settings
        self.imported_models = imported_models
        self.vllm_analyzer = None
        self.tone_analyzer = None
        self.brand_voice_validator = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize copywriting analysis components"""
        try:
            # Prioritize HybridToneAnalyzer (VLLM + OpenRouter fallback)
            if 'HybridToneAnalyzer' in self.imported_models and self.imported_models['HybridToneAnalyzer']:
                self.vllm_analyzer = self.imported_models['HybridToneAnalyzer']()
                logger.info("✅ HybridToneAnalyzer initialized with VLLM + OpenRouter fallback")
            elif 'VLLMToneAnalyzer' in self.imported_models and self.imported_models['VLLMToneAnalyzer']:
                self.vllm_analyzer = self.imported_models['VLLMToneAnalyzer']()
                logger.info("✅ VLLMToneAnalyzer initialized with real model")
            else:
                logger.warning("⚠️ VLLMToneAnalyzer not available, trying fallback models")
                self.vllm_analyzer = None
            
            # Fallback to old ToneAnalyzer if VLLM not available
            if self.vllm_analyzer is None:
                if 'ToneAnalyzer' in self.imported_models and self.imported_models['ToneAnalyzer']:
                    self.tone_analyzer = self.imported_models['ToneAnalyzer']()
                    logger.info("✅ ToneAnalyzer initialized with real model (fallback)")
                else:
                    logger.warning("⚠️ ToneAnalyzer not available, using fallback")
                    self.tone_analyzer = None
            
            # Fallback to old BrandVoiceValidator if VLLM not available
            if self.vllm_analyzer is None:
                if 'BrandVoiceValidator' in self.imported_models and self.imported_models['BrandVoiceValidator']:
                    self.brand_voice_validator = self.imported_models['BrandVoiceValidator']()
                    logger.info("✅ BrandVoiceValidator initialized with real model (fallback)")
                else:
                    logger.warning("⚠️ BrandVoiceValidator not available, using fallback")
                    self.brand_voice_validator = None
                
        except Exception as e:
            logger.error(f"Copywriting analysis initialization failed: {e}")
            import traceback
            logger.error(f"Copywriting initialization traceback: {traceback.format_exc()}")
    
    def _get_default_user_settings(self) -> Dict[str, Any]:
        """Get default user settings for VLLM analysis"""
        return {
            'formality_score': 50,
            'confidence_level': 'balanced',
            'warmth_score': 50,
            'energy_score': 50,
            'readability_level': 'grade8',
            'persona_type': 'general',
            'allow_emojis': False,
            'allow_slang': False
        }

    def _clean_text_for_llm(self, text_content: str) -> str:
        """
        Clean OCR text BEFORE sending to LLM:
        - Remove numbers
        - Remove currency symbols
        - Remove dates
        - Remove UI labels
        - Remove all-caps <=4 chars
        """
        if not text_content or not isinstance(text_content, str):
            return ""

        try:
            from ..utils.spell_checker import UI_WHITELIST
        except Exception:
            UI_WHITELIST = set()

        currency_symbols = {'$', '€', '£', '¥', '₹', '₽'}
        date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$',  # 12/31/2024
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$'        # 2024-12-31
        ]

        cleaned_tokens = []
        for token in text_content.split():
            stripped = token.strip(".,!?;:()[]{}\"'`")
            if not stripped:
                continue

            # Remove numbers (contains digits)
            if any(char.isdigit() for char in stripped):
                continue

            # Remove currency symbols
            if any(sym in stripped for sym in currency_symbols):
                continue

            # Remove dates
            if any(re.match(pattern, stripped) for pattern in date_patterns):
                continue

            # Remove all-caps <=4 chars
            if stripped.isupper() and len(stripped) <= 4:
                continue

            # Remove UI labels
            if stripped.lower() in UI_WHITELIST:
                continue

            cleaned_tokens.append(stripped)

        return " ".join(cleaned_tokens)
    
    def analyze_copywriting(self, image: np.ndarray, text_content: Optional[str] = None, ocr_tokens: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive copywriting analysis
        
        Args:
            image: Input image as numpy array
            text_content: Extracted OCR text content to analyze (REQUIRED, not config dict)
            
        Returns:
            Dictionary containing copywriting analysis results with status enum
        """
        try:
            # Only marketing-copy blocks are allowed
            if not isinstance(text_content, list):
                return {"status": "not_applicable", "issueCount": 0, "issues": []}
            return self.analyze_marketing_copy(text_content)
            
            # Use VLLM analyzer for comprehensive analysis if available
            if self.vllm_analyzer:
                try:
                        # Analyze image directly
                        user_settings = self._get_default_user_settings()
                        # Save image temporarily for VLLM analysis
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            import cv2
                            cv2.imwrite(tmp_file.name, image)
                            vllm_analysis = self.vllm_analyzer.analyze_image(tmp_file.name, user_settings)
                            print(vllm_analysis)
                        
                        if vllm_analysis:
                            # Map VLLM/Hybrid response structure to expected format
                            analysis = vllm_analysis.get('analysis', {})
                            backend_used = vllm_analysis.get('backend_used', 'unknown')
                            
                            # Extract tone analysis from nested structure
                            results['tone_analysis'] = {
                                'formality': analysis.get('formality', {}),
                                'sentiment': analysis.get('sentiment', {}),
                                'readability': analysis.get('readability', {})
                            }
                            
                            # Extract grammar analysis from nested structure
                            grammar_analysis = analysis.get('grammar', {})
                            
                            # CRITICAL: Analyze text structure and detect spelling errors
                            # This catches OCR misspellings like "Enre" → "Entire"
                            if text_content:
                                from ..utils.spell_checker import check_spelling
                                from ..utils.text_structure_analyzer import identify_text_structure, get_text_component_for_word
                                
                                # Step 1: Identify text structure (headline, subtext, CTA)
                                text_structure = identify_text_structure(clean_text or "")
                                logger.info(f"📝 Text structure identified: headline={bool(text_structure.get('headline'))}, subtext={bool(text_structure.get('subtext'))}, cta={bool(text_structure.get('cta'))}")
                                
                                # Step 2: Detect spelling errors with dictionary validation and edit-distance ≤2
                                # 🔥 Pass brand_name to protect brand names from false corrections
                                brand_name = self.settings.brand_name if hasattr(self.settings, 'brand_name') else None
                                spelling_errors, spelling_suggestions, phrase_violations = check_spelling(clean_text, brand_name=brand_name)
                                
                                # Helper function for generating explanations
                                def _generate_spelling_explanation(word: str, correction: Optional[str], severity: str, location: Optional[str], error_classification: str = 'confirmed_spelling_error') -> str:
                                    """Generate clear explanation for spelling violation"""
                                    location_str = location or 'text'
                                    
                                    # ✅ NEW: Handle OCR artifacts with clear messaging
                                    if error_classification == 'likely_ocr_artifact':
                                        if correction:
                                            return f"Likely OCR typo: '{word}' → '{correction}'. This may be an OCR artifact rather than a real spelling error."
                                        else:
                                            return f"Likely OCR artifact: '{word}' in {location_str}. This may be OCR noise rather than a real spelling error."
                                    
                                    # Confirmed spelling errors
                                    if severity == 'CRITICAL':
                                        if correction:
                                            return f"CRITICAL: Spelling error '{word}' in {location_str} should be '{correction}'. Headline errors are critical and cause compliance failure."
                                        else:
                                            return f"CRITICAL: Unknown word '{word}' in {location_str}. Headline errors are critical and cause compliance failure."
                                    elif severity == 'HIGH':
                                        if correction:
                                            return f"HIGH: Spelling error '{word}' in {location_str} (CTA) should be '{correction}'. CTA errors impact user actions."
                                        else:
                                            return f"HIGH: Unknown word '{word}' in {location_str} (CTA). CTA errors impact user actions."
                                    else:
                                        if correction:
                                            return f"{severity}: Spelling error '{word}' in {location_str} should be '{correction}'."
                                        else:
                                            return f"{severity}: Unknown word '{word}' in {location_str}."
                                
                                # Step 3: Classify errors by severity and location
                                spelling_violations = []
                                critical_errors = []
                                
                                for error in spelling_errors:
                                    if isinstance(error, dict):
                                        # 🔥 CRITICAL: Handle suspected_ocr_artifact tokens (filtered, no confidence)
                                        if error.get('suspected_ocr_artifact', False):
                                            # Filtered token - mark as suspected artifact, skip spell-check
                                            logger.debug(f"🔍 Skipping suspected OCR artifact: '{error.get('word', '')}' - {error.get('filterReason', 'unknown')}")
                                            continue  # Don't process filtered tokens
                                        
                                        error_word = error.get('word', '')
                                        correction = error.get('correction') or error.get('suggestion', '')
                                        distance = error.get('distance', -1)
                                        error_type = error.get('issueType') or error.get('type', 'misspelling')
                                        # ✅ NEW: Get error classification (confirmed_spelling_error, likely_ocr_artifact, ui_fragment)
                                        error_classification = error.get('errorType', 'confirmed_spelling_error')
                                        # ✅ Use languageConfidence (NOT OCR confidence)
                                        language_confidence = error.get('languageConfidence')
                                    else:
                                        error_word = str(error)
                                        correction = None
                                        distance = -1
                                        error_type = 'misspelling'
                                        error_classification = 'confirmed_spelling_error'  # Default if not classified
                                        language_confidence = None
                                    
                                    # Determine which component contains this word
                                    component = get_text_component_for_word(error_word, text_structure)
                                    
                                    # Determine severity - only for confirmed errors
                                    if error_classification == 'likely_ocr_artifact':
                                        severity = 'LOW'  # OCR artifacts are low severity
                                        logger.debug(f"🔍 OCR artifact detected: '{error_word}' → '{correction}' (may be OCR noise)")
                                    elif component == 'headline':
                                        severity = 'CRITICAL'
                                        critical_errors.append(f"Spelling error in headline: '{error_word}' → '{correction}'" if correction else f"Spelling error in headline: '{error_word}'")
                                        logger.warning(f"🚨 CRITICAL: Spelling error '{error_word}' detected in headline")
                                    elif component == 'cta':
                                        severity = 'HIGH'
                                    elif component == 'subtext':
                                        severity = 'MEDIUM'
                                    else:
                                        severity = 'LOW'
                                    
                                    # Build violation object with classification
                                    violation = {
                                        'word': error_word,
                                        'correction': correction,
                                        'severity': severity,
                                        'location': component or 'body',
                                        'distance': distance,
                                        'type': error_type,
                                        'issueType': error_type,  # ✅ Use issueType
                                        'errorType': error_classification,  # ✅ NEW: confirmed_spelling_error, likely_ocr_artifact, ui_fragment
                                        'languageConfidence': language_confidence,  # ✅ Language confidence, NOT OCR
                                        'suspected_ocr_artifact': error_classification == 'likely_ocr_artifact',  # ✅ Backward compatibility
                                        'explanation': _generate_spelling_explanation(error_word, correction, severity, component, error_classification)
                                    }
                                    
                                    spelling_violations.append(violation)
                                
                                # Merge spelling errors into grammar analysis
                                if spelling_errors or spelling_violations:
                                    # Initialize grammar_analysis if it doesn't exist
                                    if not grammar_analysis:
                                        grammar_analysis = {}
                                    
                                    # Add spelling errors to grammar errors
                                    existing_errors = grammar_analysis.get('grammarErrors', [])
                                    if not isinstance(existing_errors, list):
                                        existing_errors = []
                                    
                                    # Add spelling error messages with severity
                                    for violation in spelling_violations:
                                        error_msg = f"[{violation['severity']}] Spelling error in {violation['location']}: '{violation['word']}'"
                                        if violation['correction']:
                                            error_msg += f" → '{violation['correction']}'"
                                        existing_errors.append(error_msg)
                                    
                                    grammar_analysis['grammarErrors'] = existing_errors
                                    # 🔥 Return structured spelling errors, not strings
                                    grammar_analysis['spellingErrors'] = [
                                        {
                                            'word': v.get('word', ''),
                                            'suggestion': v.get('correction', ''),
                                            'confidence': v.get('confidence', 0.8),
                                            'type': v.get('type', 'unknown'),
                                            'severity': 'high' if v.get('type') == 'ocr_typo' else 'medium'
                                        }
                                        for v in spelling_violations if isinstance(v, dict)
                                    ]
                                    grammar_analysis['spellingViolations'] = spelling_violations  # Keep for backward compatibility
                                    
                                    # Add spelling suggestions to grammar suggestions
                                    existing_suggestions = grammar_analysis.get('grammarSuggestions', [])
                                    if not isinstance(existing_suggestions, list):
                                        existing_suggestions = []
                                    existing_suggestions.extend(spelling_suggestions)
                                    grammar_analysis['grammarSuggestions'] = existing_suggestions
                                    
                                    # 🔥 CRITICAL: Proportional penalty for CONFIRMED spelling errors only
                                    # Filter out likely OCR artifacts and UI fragments
                                    confirmed_spelling_violations = [
                                        v for v in spelling_violations 
                                        if isinstance(v, dict) and v.get('errorType') == 'confirmed_spelling_error'
                                    ]
                                    
                                    # Count OCR artifacts for reporting (but don't penalize)
                                    ocr_artifacts = [
                                        v for v in spelling_violations
                                        if isinstance(v, dict) and v.get('errorType') == 'likely_ocr_artifact'
                                    ]
                                    
                                    # Penalize grammar score for CONFIRMED spelling errors only
                                    grammar_score = grammar_analysis.get('grammarScore', 50)
                                    if isinstance(grammar_score, (int, float)):
                                        # 🔥 Proportional penalty: min(len(confirmed_errors) * 2, 20)
                                        # OCR artifacts do NOT reduce score
                                        penalty = min(20, len(confirmed_spelling_violations) * 2)
                                        grammar_score = max(60, grammar_score - penalty)  # Never below 60
                                        
                                        # CRITICAL: If spelling error in headline, reduce score but don't zero it
                                        if critical_errors:
                                            grammar_score = min(grammar_score, 70)  # Max 70 if headline has spelling error (not 30)
                                            grammar_analysis['critical_errors'] = critical_errors
                                            # Don't set compliance_failed - just reduce score
                                            logger.warning(f"⚠️ Headline spelling error detected - reduced grammar score to {grammar_score}")
                                        
                                        grammar_analysis['grammarScore'] = int(grammar_score)
                                    
                                    logger.info(f"🔍 Spell check found {len(confirmed_spelling_violations)} confirmed errors, {len(ocr_artifacts)} likely OCR artifacts ({len(critical_errors)} CRITICAL in headline), adjusted grammar score to {grammar_score}")
                                    
                                    # Add OCR artifact summary to grammar analysis
                                    if ocr_artifacts:
                                        grammar_analysis['ocrArtifacts'] = [
                                            {
                                                'word': v.get('word', ''),
                                                'suggestion': v.get('correction', ''),
                                                'explanation': f"Likely OCR typo: '{v.get('word', '')}' → '{v.get('correction', '')}'"
                                            }
                                            for v in ocr_artifacts
                                        ]
                                
                                # Add text structure to results
                                results['text_structure'] = text_structure
                            
                            results['grammar_analysis'] = grammar_analysis
                            
                            # Extract visual elements from nested structure
                            results['visual_elements'] = analysis.get('visual_analysis', {})
                            
                            # Extract text metrics
                            results['text_metrics'] = {
                                'word_count': vllm_analysis.get('word_count', 0),
                                'sentence_count': vllm_analysis.get('sentence_count', 0),
                                'readability_level': analysis.get('readability', {}).get('level', 'grade8')
                            }
                            
                            # Extract compliance (this is already at top level)
                            results['compliance'] = vllm_analysis.get('compliance', {})
                            
                            # Extract text content
                            text_content = vllm_analysis.get('text', '')
                            
                            # Log which backend was used
                            if backend_used != 'unknown':
                                logger.info(f"✅ Analysis completed using {backend_used} backend")
                            
                            # CRITICAL: If text_content was provided (from OCR), NEVER re-run OCR
                            # Only use OCR fallback if text_content was originally None/empty
                            # This prevents discarding valid OCR results
                            if not text_content or text_content.strip() == '':
                                # Only try OCR if text wasn't provided initially
                                logger.info("VLLM didn't extract text, trying OCR fallback...")
                                ocr_text = self._extract_text_from_image(image)
                                if ocr_text and ocr_text.strip():
                                    text_content = ocr_text
                                    logger.info(f"✅ OCR extracted text: '{text_content}'")
                                    
                                    # CRITICAL: Spell check OCR text immediately after extraction
                                    from ..utils.spell_checker import check_spelling
                                    # 🔥 Pass brand_name to protect brand names from false corrections
                                    brand_name = self.settings.brand_name if hasattr(self.settings, 'brand_name') else None
                                    spelling_errors, spelling_suggestions, phrase_violations = check_spelling(clean_text, brand_name=brand_name)
                                    
                                    # 🔥 CRITICAL: Filter REAL spelling errors only (ocr_typo, real_word_typo)
                                    # Filter out UI tokens and numeric tokens
                                    real_spelling_errors = [
                                        err for err in spelling_errors 
                                        if isinstance(err, dict) and err.get('type') in ['ocr_typo', 'real_word_typo']
                                    ]
                                    
                                    # Update grammar analysis with spelling errors
                                    if real_spelling_errors and 'grammar_analysis' in results:
                                        grammar_analysis = results.get('grammar_analysis', {})
                                        
                                        # Add spelling errors
                                        existing_errors = grammar_analysis.get('grammarErrors', [])
                                        if not isinstance(existing_errors, list):
                                            existing_errors = []
                                        for err in real_spelling_errors:
                                            word = err.get('word', '') if isinstance(err, dict) else str(err)
                                            existing_errors.append(f"Spelling error: '{word}'")
                                        grammar_analysis['grammarErrors'] = existing_errors
                                        # 🔥 Return structured spelling errors, not strings
                                        grammar_analysis['spellingErrors'] = [
                                            {
                                                'word': err.get('word', ''),
                                                'suggestion': err.get('correction', ''),
                                                'confidence': err.get('confidence', 0.8),
                                                'type': err.get('type', 'unknown'),
                                                'severity': 'high' if err.get('type') == 'ocr_typo' else 'medium'
                                            }
                                            for err in real_spelling_errors if isinstance(err, dict)
                                        ]
                                        
                                        # Add suggestions
                                        existing_suggestions = grammar_analysis.get('grammarSuggestions', [])
                                        if not isinstance(existing_suggestions, list):
                                            existing_suggestions = []
                                        existing_suggestions.extend(spelling_suggestions)
                                        grammar_analysis['grammarSuggestions'] = existing_suggestions
                                        
                                        # 🔥 Proportional penalty: min(len(real_errors) * 2, 20), never below 60
                                        grammar_score = grammar_analysis.get('grammarScore', 50)
                                        if isinstance(grammar_score, (int, float)):
                                            penalty = min(20, len(real_spelling_errors) * 2)
                                            grammar_score = max(60, grammar_score - penalty)  # Never below 60
                                            grammar_analysis['grammarScore'] = int(grammar_score)
                                        
                                        results['grammar_analysis'] = grammar_analysis
                                        logger.info(f"🔍 OCR spell check found {len(real_spelling_errors)} real errors (filtered {len(spelling_errors) - len(real_spelling_errors)} UI/numeric), adjusted grammar score to {grammar_score}")
                            else:
                                # Text was provided - use it, don't re-run OCR
                                logger.info(f"[Copywriting] Using provided text content, skipping OCR fallback")
                            
                            logger.info("✅ Used VLLMToneAnalyzer for image analysis")
                except Exception as e:
                    logger.warning(f"VLLM analysis failed: {e}, falling back to traditional methods")
                    # CRITICAL: Only extract text if it wasn't provided (from OCR)
                    # If text_content exists, use it - don't re-run OCR
                    if text_content is None or (isinstance(text_content, str) and not text_content.strip()):
                        logger.info("[Copywriting] No text available, attempting OCR extraction in fallback...")
                        text_content = self._extract_text_from_image(image)
                    else:
                        logger.info(f"[Copywriting] Using provided text content in fallback, skipping OCR")
                    
                    # Ensure text_content is a string
                    if isinstance(text_content, dict):
                        text_content = str(text_content)
                    elif not isinstance(text_content, str):
                        text_content = str(text_content) if text_content is not None else ""
                    
                    # Analyze tone
                    tone_results = self._analyze_tone(clean_text or text_content)
                    results['tone_analysis'] = tone_results
            else:
                # Traditional analysis fallback
                # CRITICAL: Only extract text if it wasn't provided (from OCR)
                # If text_content exists, use it - don't re-run OCR
                if text_content is None or (isinstance(text_content, str) and not text_content.strip()):
                    logger.info("[Copywriting] No text available, attempting OCR extraction...")
                    text_content = self._extract_text_from_image(image)
                else:
                    logger.info(f"[Copywriting] Using provided text content, skipping OCR extraction")
                
                # Ensure text_content is a string
                if isinstance(text_content, dict):
                    text_content = str(text_content)
                elif not isinstance(text_content, str):
                    text_content = str(text_content) if text_content is not None else ""
                
                # CRITICAL: Spell check text before analysis
                if text_content:
                    from ..utils.spell_checker import check_spelling
                    # 🔥 Pass brand_name to protect brand names from false corrections
                    brand_name = self.settings.brand_name if hasattr(self.settings, 'brand_name') else None
                    spelling_errors, spelling_suggestions, phrase_violations = check_spelling(clean_text, brand_name=brand_name)
                    
                    # 🔥 CRITICAL: Filter REAL spelling errors only (ocr_typo, real_word_typo)
                    # Filter out UI tokens and numeric tokens
                    real_spelling_errors = [
                        err for err in spelling_errors 
                        if isinstance(err, dict) and err.get('type') in ['ocr_typo', 'real_word_typo']
                    ]
                    
                    # Initialize grammar analysis with spelling errors
                    if real_spelling_errors:
                        grammar_errors = [f"Spelling error: '{err.get('word', err)}'" if isinstance(err, dict) else f"Spelling error: '{err}'" for err in real_spelling_errors]
                        # 🔥 Proportional penalty: min(len(real_errors) * 2, 20), never below 60
                        penalty = min(20, len(real_spelling_errors) * 2)
                        grammar_score = max(60, 50 - penalty)  # Never below 60
                        
                        # 🔥 Return structured spelling errors
                        results['spellingErrors'] = [
                            {
                                'word': err.get('word', ''),
                                'suggestion': err.get('correction', ''),
                                'confidence': err.get('confidence', 0.8),
                                'type': err.get('type', 'unknown'),
                                'severity': 'high' if err.get('type') == 'ocr_typo' else 'medium'
                            }
                            for err in real_spelling_errors if isinstance(err, dict)
                        ]
                        
                        results['grammar_analysis'] = {
                            'grammarScore': grammar_score,
                            'grammarErrors': grammar_errors,
                            'spellingErrors': real_spelling_errors,  # Only real errors
                            'grammarSuggestions': spelling_suggestions
                        }
                        logger.info(f"🔍 Traditional fallback spell check found {len(real_spelling_errors)} real errors (filtered {len(spelling_errors) - len(real_spelling_errors)} UI/numeric), grammar score: {grammar_score}")
                
                # Analyze tone
                tone_results = self._analyze_tone(clean_text or text_content)
                results['tone_analysis'] = tone_results
            
            # Validate brand voice compliance
            if 'brand_voice_compliance' not in results:
                # Use VLLM compliance if available, otherwise calculate
                if 'compliance' in results and results['compliance']:
                    # Map VLLM compliance to brand voice compliance format
                    vllm_compliance = results['compliance']
                    results['brand_voice_compliance'] = {
                        'score': vllm_compliance.get('score', 0.5),
                        'failures': vllm_compliance.get('failures', []),
                        'explanations': vllm_compliance.get('explanations', []),
                        'failure_summary': vllm_compliance.get('failure_summary', 'Analysis complete')
                    }
                else:
                    # Calculate brand voice compliance using traditional method
                    tone_results = results.get('tone_analysis', {})
                    voice_compliance = self._validate_brand_voice(text_content, tone_results)
                    results['brand_voice_compliance'] = voice_compliance
            
            # Calculate overall copywriting score
            tone_results = results.get('tone_analysis', {})
            voice_compliance = results.get('brand_voice_compliance', {})
            copywriting_score = self._calculate_copywriting_score(tone_results, voice_compliance)
            
            # CRITICAL: If critical errors exist (e.g., spelling in headline), force compliance failure
            grammar_analysis = results.get('grammar_analysis', {})
            if grammar_analysis.get('compliance_failed') or grammar_analysis.get('critical_errors'):
                # Force copywriting score to fail (max 30)
                copywriting_score = min(copywriting_score, 30.0)
                results['compliance_failed'] = True
                results['critical_errors'] = grammar_analysis.get('critical_errors', [])
                logger.error(f"🚨 CRITICAL: Copywriting compliance FAILED due to critical errors, score capped at {copywriting_score}")
            
            results['copywriting_score'] = copywriting_score
            
            # Generate recommendations
            recommendations = self._generate_copywriting_recommendations(tone_results, voice_compliance)
            results['recommendations'] = recommendations
            
            # Add extracted text to results
            results['extracted_text'] = text_content
            results['text_content'] = text_content  # Alias for compatibility
            
            # Add structured text analysis if available
            if 'text_structure' in results:
                # Ensure spelling violations are included in final output
                grammar_analysis = results.get('grammar_analysis', {})
                if grammar_analysis.get('spellingViolations'):
                    results['spelling_violations'] = grammar_analysis['spellingViolations']
                    # Calculate severity summary
                    severity_counts = {}
                    for violation in grammar_analysis['spellingViolations']:
                        severity = violation.get('severity', 'LOW')
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    results['severity_summary'] = severity_counts
            
            # Add word count to results if not already present
            if 'word_count' not in results:
                results['word_count'] = word_count
            
            # CRITICAL: Ensure status is 'success' or 'completed' when text exists and analysis ran
            # Status should only be 'failed' if no text exists OR critical system error occurred
            if has_text and results.get('status') != 'failed':
                # Text exists and analysis completed - set to success
                results['status'] = 'success'  # Changed from 'completed' to 'success' for clarity
                results['analyzerStatus'] = 'success'  # Ensure analyzerStatus matches
            
            logger.info(f"✅ Copywriting analysis completed. Score: {copywriting_score:.2f}, Status: {results.get('status', 'unknown')}")
            if text_content:
                logger.info(f"📝 Extracted text: '{text_content[:100]}...' ({word_count} words)")
            
            # ✅ Format output for human consumption (hides technical details)
            try:
                from .copywriting_output_formatter import CopywritingOutputFormatter
                # Get OCR result if available
                ocr_result = results.get('ocr_result')
                formatted_output = CopywritingOutputFormatter.format_output(
                    results, text_content, ocr_result
                )
                # Merge formatted output into results (preserve raw for internal use)
                results['formatted_output'] = formatted_output
            except Exception as e:
                logger.warning(f"Failed to format copywriting output: {e}")
            
            return results
            
        except Exception:
            return {"status": "not_applicable", "issueCount": 0, "issues": []}

    def analyze_marketing_copy(self, marketing_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze marketing copy only.
        Output schema:
        {
          status: "clean" | "needs_revision",
          issues: [{type, original, suggestion, confidence}]
        }
        """
        text_parts = []
        for block in marketing_blocks:
            text = (block.get("text") or "").strip()
            if text:
                text_parts.append(text)

        combined_text = " ".join(text_parts).strip()
        if not combined_text:
            return {"status": "not_applicable", "issueCount": 0, "issues": []}

        tokens = [t for t in combined_text.split() if any(ch.isalpha() for ch in t)]
        clean_text = " ".join(tokens).strip()
        if not clean_text:
            return {"status": "not_applicable", "issueCount": 0, "issues": []}

        issues = []
        try:
            from ..utils.spell_checker import check_spelling
            brand_name = self.settings.brand_name if hasattr(self.settings, 'brand_name') else None
            spelling_errors, _, _ = check_spelling(clean_text, brand_name=brand_name)
            seen = set()
            for error in spelling_errors:
                if not isinstance(error, dict):
                    continue
                if error.get("suspected_ocr_artifact"):
                    continue
                original = error.get("word", "")
                suggestion = error.get("suggestion") or error.get("correction") or ""
                if not original:
                    continue
                key = (original.lower(), suggestion.lower())
                if key in seen:
                    continue
                seen.add(key)
                lang_conf = error.get("languageConfidence")
                confidence_level = "high" if isinstance(lang_conf, (int, float)) and lang_conf >= 0.85 else "medium"
                issues.append({
                    "type": "spelling",
                    "original": original,
                    "suggestion": suggestion,
                    "confidence": confidence_level
                })
        except Exception:
            issues = []

        status = "needs_revision" if issues else "clean"
        return {"status": status, "issueCount": len(issues), "issues": issues}
    
    def _extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text content from image using OCR with preprocessing"""
        from ..utils.ocr_utils import preprocess_image_for_ocr
        
        ocr_methods_tried = []
        
        # STEP 1: Preprocess image for OCR (BGR→RGB, 2× upscale)
        ocr_image, preprocess_error = preprocess_image_for_ocr(image)
        if ocr_image is None:
            logger.warning(f"[Copywriting] OCR preprocessing failed: {preprocess_error}")
            return ""  # Return empty string, don't fail
        
        try:
            # Try using pytesseract first (most common OCR library)
            try:
                import pytesseract
                ocr_methods_tried.append('pytesseract')
                # Use preprocessed RGB image (already converted and upscaled)
                from PIL import Image
                pil_image = Image.fromarray(ocr_image)
                
                # Perform OCR
                extracted_text = pytesseract.image_to_string(pil_image, lang='eng')
                # CRITICAL: Accept text even with low confidence - preserve partial text
                if extracted_text and extracted_text.strip():
                    logger.info(f"✅ OCR extracted {len(extracted_text.split())} words using pytesseract")
                    return extracted_text.strip()
                else:
                    logger.warning("pytesseract returned empty text, trying fallback methods")
            except ImportError as e:
                logger.warning(f"pytesseract not installed: {e}")
            except Exception as e:
                # Check for common tesseract issues
                error_str = str(e).lower()
                if 'tesseract is not installed' in error_str or 'tesseract-ocr' in error_str:
                    logger.error("❌ Tesseract OCR binary not installed. Install with: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)")
                else:
                    logger.warning(f"pytesseract OCR failed: {e}")
            
            # Try PaddleOCR as fallback
            try:
                from paddleocr import PaddleOCR
                ocr_methods_tried.append('PaddleOCR')
                # Initialize PaddleOCR (lazy initialization)
                if not hasattr(self, '_paddle_ocr'):
                    logger.info("Initializing PaddleOCR...")
                    self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                
                # Use preprocessed RGB image (already converted and upscaled)
                ocr_result = self._paddle_ocr.ocr(ocr_image, cls=True)
                if ocr_result and ocr_result[0]:
                    # Extract text from all detected regions
                    text_lines = [line[1][0] for line in ocr_result[0] if line and len(line) > 1]
                    extracted_text = '\n'.join(text_lines)
                    # CRITICAL: Accept text even with low confidence - preserve partial text
                    if extracted_text and extracted_text.strip():
                        logger.info(f"✅ OCR extracted {len(extracted_text.split())} words using PaddleOCR")
                        return extracted_text.strip()
                    else:
                        logger.warning("PaddleOCR returned no text regions")
            except ImportError as e:
                logger.warning(f"PaddleOCR not installed: {e}")
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")
            
            # Try VLLM analyzer if available (last resort)
            if self.vllm_analyzer:
                try:
                    import tempfile
                    import os
                    ocr_methods_tried.append('VLLM')
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        cv2.imwrite(tmp_file.name, image)
                        tmp_path = tmp_file.name
                    
                    try:
                        vllm_result = self.vllm_analyzer.analyze_image(tmp_path, {
                            'task': 'ocr',
                            'prompt': 'Extract all visible text from this image. Return only the text, no analysis.'
                        })
                        if vllm_result and vllm_result.get('text'):
                            extracted_text = vllm_result['text']
                            if extracted_text and extracted_text.strip():
                                logger.info(f"✅ OCR extracted {len(extracted_text.split())} words using VLLM")
                                return extracted_text.strip()
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"VLLM text extraction failed: {e}")
            else:
                logger.warning("VLLM analyzer not available for OCR fallback")
            
            # If all OCR methods failed, return empty string with diagnostic info
            logger.warning(f"⚠️ All OCR methods failed. Methods tried: {ocr_methods_tried}")
            logger.warning("To fix OCR: Install tesseract-ocr binary AND pytesseract, or install paddleocr")
            
            # Log detailed diagnostics for troubleshooting
            logger.error("OCR DIAGNOSTICS:")
            logger.error(f"  - Image shape: {image.shape if isinstance(image, np.ndarray) else 'unknown'}")
            logger.error(f"  - Image dtype: {image.dtype if isinstance(image, np.ndarray) else 'unknown'}")
            if isinstance(image, np.ndarray):
                logger.error(f"  - Image min/max values: {image.min()}/{image.max()}")
                logger.error(f"  - Image mean: {image.mean():.2f}")
            
            # Try one more time with additional image enhancement (on already preprocessed image)
            try:
                logger.info("Attempting OCR with additional image enhancement...")
                # ocr_image is already RGB and upscaled, now enhance further
                # Convert to grayscale for thresholding
                gray = cv2.cvtColor(ocr_image, cv2.COLOR_RGB2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply adaptive thresholding for better contrast
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                
                # Try pytesseract one more time with enhanced image
                try:
                    import pytesseract
                    from PIL import Image
                    pil_image = Image.fromarray(thresh)
                    extracted_text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6')
                    # CRITICAL: Accept text even with low confidence - preserve partial text
                    if extracted_text and extracted_text.strip() and len(extracted_text.strip()) > 2:
                        logger.info(f"✅ OCR succeeded with additional enhancement: {len(extracted_text.split())} words")
                        return extracted_text.strip()
                except Exception as e:
                    logger.debug(f"Enhanced OCR attempt failed: {e}")
            except Exception as e:
                logger.debug(f"Image preprocessing failed: {e}")
            
            return ""
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            import traceback
            logger.error(f"OCR traceback: {traceback.format_exc()}")
            return ""
    
    def _analyze_tone(self, text_content: str) -> Dict[str, Any]:
        """Analyze the tone of the text content"""
        try:
            tone_results = {
                'detected_tone': 'neutral',
                'confidence': 0.5,
                'tone_attributes': {},
                'sentiment_score': 0.0
            }
            
            if self.vllm_analyzer and text_content:
                # Use VLLM/Hybrid analyzer for tone analysis
                try:
                    user_settings = self._get_default_user_settings()
                    cleaned_text = self._clean_text_for_llm(text_content)
                    if not cleaned_text:
                        logger.warning("[Copywriting] Cleaned text is empty after OCR noise filtering; skipping LLM tone analysis")
                        return tone_results
                    analysis = self.vllm_analyzer.analyze_text(cleaned_text, user_settings)
                    if analysis and 'analysis' in analysis:
                        tone_results.update(analysis['analysis'])
                        backend_used = analysis.get('backend_used', 'unknown')
                        logger.info(f"✅ Used analyzer for tone analysis (backend: {backend_used})")
                    elif analysis and 'tone_analysis' in analysis:
                        tone_results.update(analysis['tone_analysis'])
                        logger.info("✅ Used analyzer for tone analysis")
                except Exception as e:
                    logger.warning(f"VLLM/Hybrid tone analysis failed: {e}, falling back to old analyzer")
                    if self.tone_analyzer:
                        cleaned_text = self._clean_text_for_llm(text_content)
                        if not cleaned_text:
                            logger.warning("[Copywriting] Cleaned text is empty after OCR noise filtering; skipping fallback tone analysis")
                            return tone_results
                        analysis = self.tone_analyzer.analyze_text_tone(cleaned_text)
                        if analysis:
                            tone_results.update(analysis)
            elif self.tone_analyzer and text_content:
                # Use old tone analyzer as fallback
                cleaned_text = self._clean_text_for_llm(text_content)
                if not cleaned_text:
                    logger.warning("[Copywriting] Cleaned text is empty after OCR noise filtering; skipping fallback tone analysis")
                    return tone_results
                analysis = self.tone_analyzer.analyze_text_tone(cleaned_text)
                if analysis:
                    tone_results.update(analysis)
                logger.info("✅ Used ToneAnalyzer (fallback) for tone analysis")
            else:
                # Fallback tone analysis
                tone_results = self._fallback_tone_analysis(text_content)
                logger.info("✅ Used fallback tone analysis")
            
            return tone_results
            
        except Exception as e:
            logger.error(f"Tone analysis failed: {e}")
            return {
                'detected_tone': 'unknown',
                'confidence': 0.0,
                'tone_attributes': {},
                'sentiment_score': 0.0
            }
    
    def _validate_brand_voice(self, text_content: str, tone_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text against brand voice guidelines"""
        try:
            compliance_results = {
                'is_compliant': False,
                'compliance_score': 0.0,
                'violations': [],
                'strengths': []
            }
            
            if self.brand_voice_validator and text_content:
                # Use real brand voice validator - correct method name is validate_brand_voice
                # Create a default brand profile
                brand_profile = {
                    'tone_preference': 'professional',
                    'formality_preference': 'formal',
                    'emotion_preference': 'neutral'
                }
                validation = self.brand_voice_validator.validate_brand_voice(text_content, brand_profile)
                if validation:
                    compliance_results.update(validation)
            else:
                # Fallback brand voice validation
                compliance_results = self._fallback_brand_voice_validation(text_content, tone_results)
            
            return compliance_results
            
        except Exception as e:
            logger.error(f"Brand voice validation failed: {e}")
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'violations': ['Validation failed due to technical error'],
                'strengths': []
            }
    
    def _calculate_copywriting_score(self, tone_results: Dict[str, Any], voice_compliance: Dict[str, Any]) -> float:
        """Calculate overall copywriting score"""
        try:
            # Base score from brand voice compliance
            base_score = voice_compliance.get('score', voice_compliance.get('compliance_score', 0.0))
            
            # Extract tone confidence from nested structure
            formality = tone_results.get('formality', {})
            sentiment = tone_results.get('sentiment', {})
            
            # Tone confidence factor (use formality score as confidence)
            tone_confidence = formality.get('formality_score', 0.5)
            tone_factor = tone_confidence * 0.3
            
            # Sentiment score factor (convert sentiment to numeric)
            sentiment_text = sentiment.get('overall_sentiment', 'neutral')
            if sentiment_text == 'positive':
                sentiment_score = 0.8
            elif sentiment_text == 'negative':
                sentiment_score = 0.2
            else:
                sentiment_score = 0.5
            sentiment_factor = sentiment_score * 0.2
            
            # Calculate final score
            final_score = base_score * 0.5 + tone_factor + sentiment_factor
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Copywriting score calculation failed: {e}")
            return 0.0
    
    def _generate_copywriting_recommendations(self, tone_results: Dict[str, Any], voice_compliance: Dict[str, Any]) -> List[str]:
        """Generate copywriting recommendations"""
        try:
            recommendations = []
            
            # Tone-based recommendations
            detected_tone = tone_results.get('detected_tone', 'unknown')
            confidence = tone_results.get('confidence', 0.0)
            
            if confidence < 0.7:
                recommendations.append("Improve tone clarity for better brand voice alignment")
            
            # Brand voice compliance recommendations
            violations = voice_compliance.get('violations', [])
            if violations:
                recommendations.append(f"Address {len(violations)} brand voice violations")
            
            compliance_score = voice_compliance.get('compliance_score', 0.0)
            if compliance_score < 0.8:
                recommendations.append("Enhance brand voice compliance for better consistency")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Copywriting meets brand voice standards")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Copywriting recommendations generation failed: {e}")
            return ["Review copywriting for brand voice compliance"]
    
    def _fallback_tone_analysis(self, text_content: str) -> Dict[str, Any]:
        """Fallback tone analysis when real model is not available"""
        try:
            if not text_content:
                return {
                    'detected_tone': 'neutral',
                    'confidence': 0.0,
                    'tone_attributes': {},
                    'sentiment_score': 0.0
                }
            
            # Basic keyword-based tone detection
            positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
            
            text_lower = text_content.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                tone = 'positive'
                sentiment = 0.5
            elif negative_count > positive_count:
                tone = 'negative'
                sentiment = -0.5
            else:
                tone = 'neutral'
                sentiment = 0.0
            
            return {
                'detected_tone': tone,
                'confidence': 0.6,
                'tone_attributes': {'word_count': len(text_content.split())},
                'sentiment_score': sentiment
            }
            
        except Exception as e:
            logger.error(f"Fallback tone analysis failed: {e}")
            return {
                'detected_tone': 'neutral',
                'confidence': 0.0,
                'tone_attributes': {},
                'sentiment_score': 0.0
            }
    
    def _fallback_brand_voice_validation(self, text_content: str, tone_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback brand voice validation when real model is not available"""
        try:
            # Basic validation based on text length and tone
            violations = []
            strengths = []
            
            if len(text_content) < 10:
                violations.append("Text content too short for effective communication")
            else:
                strengths.append("Adequate text length for communication")
            
            tone = tone_results.get('detected_tone', 'neutral')
            if tone == 'neutral':
                strengths.append("Neutral tone is generally safe for brand voice")
            
            compliance_score = 0.7 if not violations else 0.3
            
            return {
                'is_compliant': len(violations) == 0,
                'compliance_score': compliance_score,
                'violations': violations,
                'strengths': strengths
            }
            
        except Exception as e:
            logger.error(f"Fallback brand voice validation failed: {e}")
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'violations': ['Validation failed due to technical error'],
                'strengths': []
            }
    
    def cleanup(self):
        """Clean up resources to prevent memory leaks"""
        try:
            logger.info("Cleaning up copywriting analyzer...")
            
            # Clear model references
            if hasattr(self, 'tone_analyzer') and self.tone_analyzer:
                if hasattr(self.tone_analyzer, 'cleanup'):
                    self.tone_analyzer.cleanup()
                del self.tone_analyzer
                self.tone_analyzer = None
            
            if hasattr(self, 'voice_validator') and self.voice_validator:
                if hasattr(self.voice_validator, 'cleanup'):
                    self.voice_validator.cleanup()
                del self.voice_validator
                self.voice_validator = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Copywriting analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during copywriting analyzer cleanup: {e}")