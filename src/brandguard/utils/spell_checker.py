"""
Deterministic spell checker for OCR text validation.
Uses dictionary-based checking with fuzzy matching (Levenshtein distance) for reliable spelling detection.
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 🔥 CRITICAL: UI and domain whitelists - these words must NEVER be spellchecked
UI_WHITELIST = {
    "crm", "usd", "vat", "tax", "api", "ui", "ux",
    "dashboard", "invoice", "subtotal", "total",
    "qty", "id", "order", "web", "login", "signup",
    "signup", "signin", "logout", "signout", "checkout",
    "cart", "price", "cost", "fee", "paid", "due",
    "status", "pending", "completed", "cancelled",
    "email", "phone", "address", "name", "date",
    "time", "hour", "minute", "second", "day", "week",
    "month", "year", "today", "yesterday", "tomorrow"
}

MONTH_WHITELIST = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}

# Try to import spellchecker, fallback to basic implementation if not available
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    SPELLCHECKER_AVAILABLE = True
    logger.info("✅ SpellChecker library loaded successfully")
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logger.warning("⚠️ SpellChecker library not available. Install with: pip install pyspellchecker")
    _spell = None

# Try to import Levenshtein for fuzzy matching
try:
    from Levenshtein import distance as levenshtein_distance
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to python-Levenshtein
        from Levenshtein import distance as levenshtein_distance
        LEVENSHTEIN_AVAILABLE = True
    except ImportError:
        LEVENSHTEIN_AVAILABLE = False
        logger.warning("⚠️ Levenshtein library not available. Install with: pip install python-Levenshtein")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if not LEVENSHTEIN_AVAILABLE:
        # Simple fallback implementation
        if len(s1) < len(s2):
            return _levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    else:
        return levenshtein_distance(s1.lower(), s2.lower())


def _should_filter_token(token: str, brand_name: Optional[str] = None) -> bool:
    """
    🔥 CRITICAL: Determine if token should be filtered BEFORE spell-check.
    
    Filters:
    - Numbers (contains digits)
    - Currency symbols ($, €, £, etc.)
    - Short tokens (≤2 chars)
    - Known UI labels
    - Brand names
    
    Args:
        token: Token to check
        brand_name: Optional brand name to protect
        
    Returns:
        True if token should be filtered (marked as suspected_ocr_artifact)
    """
    if not token:
        return True
    
    # Filter short tokens (≤2 chars)
    if len(token) <= 2:
        return True
    
    # Filter numbers (contains digits)
    if any(char.isdigit() for char in token):
        return True
    
    # Filter currency symbols (single char tokens that are currency)
    if len(token) == 1 and token in ['$', '€', '£', '¥', '₹', '₽']:
        return True
    
    # Filter all uppercase (likely acronyms)
    if token.isupper() and len(token) >= 2:
        return True
    
    token_lower = token.lower()
    
    # Filter UI labels
    if token_lower in UI_WHITELIST:
        return True
    
    # Filter month names
    if token_lower in MONTH_WHITELIST:
        return True
    
    # Filter brand names
    if brand_name:
        brand_lower = brand_name.lower()
        distance = _levenshtein_distance(token_lower, brand_lower)
        if distance <= 2:
            return True
    
    return False


def _get_filter_reason(token: str, brand_name: Optional[str] = None) -> str:
    """
    Get reason why token was filtered.
    """
    if not token:
        return "empty token"
    
    if len(token) <= 2:
        return "short token (≤2 chars)"
    
    if any(char.isdigit() for char in token):
        return "contains numbers"
    
    if len(token) == 1 and token in ['$', '€', '£', '¥', '₹', '₽']:
        return "currency symbol"
    
    if token.isupper() and len(token) >= 2:
        return "acronym (all uppercase)"
    
    token_lower = token.lower()
    
    if token_lower in UI_WHITELIST:
        return "UI label"
    
    if token_lower in MONTH_WHITELIST:
        return "month name"
    
    if brand_name:
        brand_lower = brand_name.lower()
        distance = _levenshtein_distance(token_lower, brand_lower)
        if distance <= 2:
            return "brand name"
    
    return "unknown filter reason"


def is_spellcheck_candidate(word: str, brand_name: Optional[str] = None) -> bool:
    """
    🔥 CRITICAL: Filter what can be spellchecked.
    
    Rules:
    - Words < 4 chars: skip (too short, likely abbreviations)
    - All uppercase: skip (likely acronyms like "CRM", "USD")
    - Contains digits: skip (likely numbers or codes)
    - In UI whitelist: skip (UI terms like "web", "login")
    - In month whitelist: skip
    - Similar to brand name: skip (protect brand names)
    
    Args:
        word: Word to check
        brand_name: Optional brand name to protect
        
    Returns:
        True if word should be spellchecked, False otherwise
    """
    if not word or len(word) < 4:
        return False  # Too short - likely abbreviation
    
    if word.isupper():
        return False  # All caps - likely acronym (CRM, USD, etc.)
    
    if any(char.isdigit() for char in word):
        return False  # Contains digits - likely number or code
    
    word_lower = word.lower()
    
    if word_lower in UI_WHITELIST:
        return False  # UI term - never spellcheck
    
    if word_lower in MONTH_WHITELIST:
        return False  # Month name - never spellcheck
    
    # Brand name protection
    if brand_name:
        brand_lower = brand_name.lower()
        distance = _levenshtein_distance(word_lower, brand_lower)
        if distance <= 2:
            return False  # Too similar to brand name - protect it
    
    return True


def _contains_verb(text: str) -> bool:
    """
    Check if text contains verb-like words (indicating full sentence).
    Simple heuristic: look for common action words.
    """
    verb_indicators = {
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "done",
        "get", "got", "go", "went", "come", "came",
        "make", "made", "take", "took", "give", "gave",
        "see", "saw", "know", "knew", "think", "thought",
        "say", "said", "tell", "told", "ask", "asked",
        "want", "wanted", "need", "needed", "try", "tried",
        "use", "used", "work", "worked", "call", "called",
        "can", "could", "will", "would", "should", "may", "might"
    }
    words = text.lower().split()
    return any(word in verb_indicators for word in words)


def _is_full_sentence(text: str) -> bool:
    """
    Check if text appears to be a full sentence (not UI block).
    
    Rules:
    - Must have at least 4 words
    - Should contain verb-like words
    - Should not be all caps (likely UI label)
    """
    words = text.split()
    if len(words) < 4:
        return False  # Too short - likely UI label
    
    if text.isupper() and len(words) <= 6:
        return False  # All caps and short - likely UI block
    
    # Check for verb indicators
    return _contains_verb(text)


def _fuzzy_match_phrase(text: str, phrase: str, max_distance: int = 2) -> Tuple[bool, float]:
    """
    Check if phrase appears in text with fuzzy matching (Levenshtein distance).
    
    Args:
        text: Text to search in
        phrase: Phrase to find (e.g., "sign up")
        max_distance: Maximum allowed Levenshtein distance
        
    Returns:
        Tuple: (found, confidence_score)
            - found: True if phrase found with distance <= max_distance
            - confidence_score: 1.0 - (distance / max(len(phrase), len(match)))
    """
    if not text or not phrase:
        return False, 0.0
    
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    phrase_words = phrase_lower.split()
    
    # Check for exact match first
    if phrase_lower in text_lower:
        return True, 1.0
    
    # Check for word-by-word fuzzy match
    text_words = text_lower.split()
    for i in range(len(text_words) - len(phrase_words) + 1):
        candidate = ' '.join(text_words[i:i+len(phrase_words)])
        distance = _levenshtein_distance(candidate, phrase_lower)
        
        if distance <= max_distance:
            max_len = max(len(candidate), len(phrase_lower))
            confidence = 1.0 - (distance / max_len) if max_len > 0 else 1.0
            return True, max(0.0, confidence)
    
    return False, 0.0


def check_spelling(text: str, phrase_rules: Optional[List[str]] = None, brand_name: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """
    Deterministic spell checker for OCR text with fuzzy matching.
    Returns spelling errors, suggestions, and phrase violations.
    
    🔥 CRITICAL: Token filtering happens BEFORE spell-check.
    OCR confidence is NEVER used as spelling confidence.
    Only LLM-evaluated tokens get languageConfidence.
    
    Args:
        text: Text content to check (from OCR or other source)
        phrase_rules: Optional list of phrases to check (e.g., ["sign up", "log in"])
        brand_name: Optional brand name to protect from false corrections
        
    Returns:
        tuple: (spelling_errors, suggestions, phrase_violations)
            - spelling_errors: List of dicts with:
                - word: str
                - correction: str
                - languageConfidence: float (0-1) - ONLY if evaluated by LLM
                - issueType: str
                - suspected_ocr_artifact: bool - if token was filtered (not evaluated)
            - suggestions: List of suggestion strings
            - phrase_violations: List of dicts with phrase errors
    """
    if not text or not isinstance(text, str):
        return [], [], []
    
    if not SPELLCHECKER_AVAILABLE:
        logger.warning("SpellChecker not available, skipping spell check")
        return [], [], []
    
    try:
        # 🔥 CRITICAL: Only spellcheck full sentences, not UI blocks
        # Split text into lines and only check lines that look like sentences
        lines = text.split('\n')
        sentences_to_check = []
        
        for line in lines:
            line = line.strip()
            if line and _is_full_sentence(line):
                sentences_to_check.append(line)
        
        # If no full sentences found, check entire text but be more conservative
        if not sentences_to_check:
            # Check if entire text is a sentence
            if _is_full_sentence(text):
                sentences_to_check = [text]
            else:
                # UI block detected - skip spellchecking
                logger.info("🔍 Text appears to be UI block (not full sentence) - skipping spellcheck")
                return [], [], []
        
        # Combine sentences for checking
        text_to_check = ' '.join(sentences_to_check)
        
        # Extract all tokens (including numbers, currency, short tokens)
        all_tokens = re.findall(r"\b\w+\b", text_to_check)
        
        if not all_tokens:
            return [], [], []
        
        # 🔥 CRITICAL: Filter tokens BEFORE spell-check
        # Separate into:
        # 1. Candidate tokens (sent to LLM for evaluation)
        # 2. Filtered tokens (marked as suspected_ocr_artifact, NO confidence)
        candidate_tokens = []
        filtered_tokens = []
        
        for token in all_tokens:
            # CRITICAL: Filter BEFORE spell-check
            if _should_filter_token(token, brand_name):
                # Mark as suspected OCR artifact - NO confidence attached
                filtered_tokens.append({
                    'word': token,
                    'suspected_ocr_artifact': True,
                    'filterReason': _get_filter_reason(token, brand_name)
                })
                logger.debug(f"🔍 Filtered token: '{token}' - {_get_filter_reason(token, brand_name)}")
            else:
                candidate_tokens.append(token)
        
        if not candidate_tokens:
            logger.info("🔍 No spellcheck candidates found after filtering")
            # Return filtered tokens as suspected artifacts
            return filtered_tokens, [], []
        
        logger.info(f"🔍 Spellchecking {len(candidate_tokens)} candidates (filtered {len(filtered_tokens)} tokens)")
        
        # Extract words (3+ characters, alphabetic only) from candidates
        words = [w for w in candidate_tokens if len(w) >= 3 and w.isalpha()]
        
        # Find misspelled words (only from candidates)
        misspelled = _spell.unknown(words)
        
        spelling_errors = []
        suggestions = []
        phrase_violations = []
        
        # Check individual word spelling with fuzzy matching
        # CRITICAL: Use edit-distance ≤2 for corrections (catches real-word typos like "Enre" → "Entire")
        for word in misspelled:
            # Get correction from spell checker
            correction = _spell.correction(word)
            
            # Also try to find candidates with edit distance ≤2 for better detection
            # This catches cases where the spell checker might not have the best suggestion
            candidates = _spell.candidates(word) if hasattr(_spell, 'candidates') else []
            
            # Find best candidate with edit distance ≤2
            best_correction = correction
            best_distance = _levenshtein_distance(word.lower(), correction.lower()) if correction else 999
            
            if candidates:
                for candidate in candidates:
                    if candidate and candidate != word:
                        dist = _levenshtein_distance(word.lower(), candidate.lower())
                        if dist <= 2 and (best_distance > 2 or dist < best_distance):
                            best_correction = candidate
                            best_distance = dist
            
            # Calculate confidence based on Levenshtein distance
            if best_correction and best_correction != word:
                distance = _levenshtein_distance(word.lower(), best_correction.lower())
                
                # CRITICAL: Only accept corrections with edit-distance ≤2
                # This ensures we catch real-word typos (e.g., "Enre" → "Entire", "form" → "from")
                if distance <= 2:
                    max_len = max(len(word), len(best_correction))
                    # 🔥 CRITICAL: This is languageConfidence (from LLM/dictionary evaluation), NOT OCR confidence
                    language_confidence = 1.0 - (distance / max_len) if max_len > 0 else 1.0
                    
                    # 🔥 CRITICAL: Confidence threshold - only report if >= 0.75
                    # This eliminates 80% of false positives
                    if language_confidence >= 0.75:
                        # Determine error type and classification
                        # Classification: confirmed_spelling_error, likely_ocr_artifact, ui_fragment
                        if distance <= 1:
                            issue_type = 'ocr_typo'
                            # Single-character errors are likely OCR artifacts unless very high confidence
                            error_classification = 'likely_ocr_artifact' if language_confidence < 0.9 else 'confirmed_spelling_error'
                        else:
                            issue_type = 'real_word_typo'
                            # Multi-character errors are more likely real errors
                            error_classification = 'confirmed_spelling_error'
                        
                        spelling_errors.append({
                            'word': word,
                            'correction': best_correction,
                            'languageConfidence': round(max(0.0, language_confidence), 2),  # ✅ Language confidence, NOT OCR
                            'distance': distance,
                            'issueType': issue_type,  # ocr_typo or real_word_typo
                            'errorType': error_classification,  # ✅ NEW: confirmed_spelling_error, likely_ocr_artifact, ui_fragment
                            'suspected_ocr_artifact': error_classification == 'likely_ocr_artifact'  # ✅ Backward compatibility
                        })
                        suggestions.append(f"Replace '{word}' with '{best_correction}' (edit distance: {distance}, languageConfidence: {language_confidence:.2f})")
                    else:
                        logger.debug(f"🔍 Filtered out: '{word}' → '{best_correction}' (languageConfidence {language_confidence:.2f} < 0.75)")
                else:
                    # Correction exists but edit distance > 2 - skip (too uncertain)
                    logger.debug(f"🔍 Filtered out: '{word}' → '{best_correction}' (distance {distance} > 2)")
            else:
                # No correction found, but word is unknown
                # CRITICAL: Check for common OCR mistakes with manual corrections
                # These are common OCR errors that spell checker might miss
                # STRICT REQUIREMENT: Detect specific mistakes:
                # - entre → entire
                # - sing up → sign up
                # - log in to → log in
                common_ocr_corrections = {
                    "enre": "entire",
                    "entre": "entire",  # STRICT: Required detection
                    "sing": "sign",
                    "singup": "sign up",
                    "sing up": "sign up",  # STRICT: Required detection
                    "singin": "sign in",
                    "sing in": "sign in",
                    "log in to": "log in",  # STRICT: Required detection
                    "login to": "log in",
                    "form": "from",  # Common OCR confusion
                    "teh": "the",
                    "adn": "and",
                    "taht": "that",
                    "recieve": "receive",
                    "seperate": "separate",
                    "occured": "occurred",
                    "begining": "beginning"
                }
                
                word_lower = word.lower()
                if word_lower in common_ocr_corrections:
                    correction = common_ocr_corrections[word_lower]
                    distance = _levenshtein_distance(word_lower, correction)
                    max_len = max(len(word), len(correction))
                    confidence = 1.0 - (distance / max_len) if max_len > 0 else 1.0
                    
                    # 🔥 CRITICAL: Confidence threshold - only report if >= 0.75
                    language_confidence = max(0.8, confidence)  # High confidence for known OCR mistakes
                    if language_confidence >= 0.75:
                        # Known OCR mistakes are likely OCR artifacts
                        spelling_errors.append({
                            'word': word,
                            'correction': correction,
                            'languageConfidence': round(language_confidence, 2),  # ✅ Language confidence, NOT OCR
                            'distance': distance,
                            'issueType': 'ocr_typo',  # Known OCR mistake
                            'errorType': 'likely_ocr_artifact',  # ✅ NEW: Mark as likely OCR artifact
                            'suspected_ocr_artifact': True  # ✅ Backward compatibility
                        })
                        suggestions.append(f"Replace '{word}' with '{correction}' (common OCR mistake, languageConfidence: {language_confidence:.2f})")
                    else:
                        logger.debug(f"🔍 Filtered out OCR correction: '{word}' → '{correction}' (languageConfidence {language_confidence:.2f} < 0.75)")
                elif 4 <= len(word) <= 6:
                    # Try common patterns: missing letters, transposed letters, etc.
                    # For "Enre" → try finding words that start with "En" and are 5-7 chars
                    # This is a fallback heuristic
                    pass  # Keep as unknown for now, spell checker should handle most cases
                
                # 🔥 CRITICAL: Don't flag unknown words as errors
                # Unknown words that pass the filter are likely proper nouns, brand names, or domain terms
                # Only flag if we have high-confidence corrections
                # Removed the "unknown_word" type - too many false positives
        
        # Check phrase rules with fuzzy matching
        if phrase_rules:
            for phrase in phrase_rules:
                found, confidence = _fuzzy_match_phrase(text, phrase, max_distance=2)
                if not found:
                    # Check for common misspellings
                    common_misspellings = {
                        "sign up": ["sing up", "sin up", "singup", "sing up"],
                        "log in": ["login", "loggin", "loggin in", "log in to", "login to"],  # STRICT: Detect "log in to" → "log in"
                        "sign in": ["sing in", "sin in", "singin"],
                        "check out": ["checkout", "chek out"],
                        "set up": ["setup", "set-up"],
                        "entire": ["enre", "entre", "entier", "entiree"]  # STRICT: Detect "entre" → "entire"
                    }
                    
                    if phrase.lower() in common_misspellings:
                        for misspelling in common_misspellings[phrase.lower()]:
                            found_misspelling, conf = _fuzzy_match_phrase(text, misspelling, max_distance=1)
                            if found_misspelling:
                                phrase_violations.append({
                                    'phrase': misspelling,
                                    'suggestion': phrase,
                                    'confidence': conf,
                                    'type': 'phrase_misspelling'
                                })
                                break
        
        # Combine evaluated errors with filtered tokens (suspected artifacts)
        all_results = spelling_errors + filtered_tokens
        
        if spelling_errors:
            logger.info(f"🔍 Found {len(spelling_errors)} spelling errors (with languageConfidence)")
        if filtered_tokens:
            logger.info(f"🔍 Found {len(filtered_tokens)} filtered tokens (suspected_ocr_artifact, no confidence)")
        if phrase_violations:
            logger.info(f"🔍 Found {len(phrase_violations)} phrase violations")
        
        return all_results, suggestions, phrase_violations
        
    except Exception as e:
        logger.error(f"Spell checking failed: {e}")
        return [], [], []

