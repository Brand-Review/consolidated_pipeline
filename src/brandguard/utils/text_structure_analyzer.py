"""
Text Structure Analyzer
Identifies headline, subtext, and CTA (Call-to-Action) from extracted text.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common CTA patterns
CTA_PATTERNS = [
    r'\b(sign up|signup|register|join now|get started|start free|try now|learn more|read more|shop now|buy now|order now|download|subscribe|claim|apply|enroll|activate|begin|explore|discover|view|see more|click here|get it|order|purchase|add to cart|checkout|check out)\b',
    r'\b(click|tap|swipe|press|select|choose|pick|grab|get|take|find|search|browse|explore|discover)\b',
    r'\b(now|today|here|free|limited|exclusive|special|offer|deal|sale|discount|save|win|prize)\b',
]

# CTA indicators (words that suggest CTA)
CTA_INDICATORS = [
    'button', 'link', 'action', 'cta', 'call-to-action', 'click', 'tap'
]


def identify_text_structure(text: str) -> Dict[str, Any]:
    """
    Identify headline, subtext, and CTA from extracted text.
    
    Args:
        text: Raw extracted text from OCR
        
    Returns:
        Dictionary with structured text components:
        {
            'headline': str or None,
            'subtext': str or None,
            'cta': str or None,
            'body': str or None,
            'structure_confidence': float
        }
    """
    if not text or not isinstance(text, str):
        return {
            'headline': None,
            'subtext': None,
            'cta': None,
            'body': None,
            'structure_confidence': 0.0
        }
    
    # Split text into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return {
            'headline': None,
            'subtext': None,
            'cta': None,
            'body': None,
            'structure_confidence': 0.0
        }
    
    headline = None
    subtext = None
    cta = None
    body_lines = []
    
    # Strategy 1: First line is often headline (if it's short and prominent)
    if lines:
        first_line = lines[0]
        # Headline characteristics: short, often all caps or title case, first line
        if len(first_line) < 100 and (first_line.isupper() or first_line.istitle() or len(first_line.split()) <= 10):
            headline = first_line
            lines = lines[1:]  # Remove headline from remaining lines
    
    # Strategy 2: Look for CTA patterns
    cta_candidates = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Check for CTA patterns
        for pattern in CTA_PATTERNS:
            if re.search(pattern, line_lower, re.IGNORECASE):
                cta_candidates.append((i, line))
                break
        
        # Check for CTA indicators
        for indicator in CTA_INDICATORS:
            if indicator in line_lower:
                cta_candidates.append((i, line))
                break
    
    # Select best CTA (usually last candidate, or shortest line)
    if cta_candidates:
        # Prefer shorter lines as CTA (buttons are usually short)
        cta_candidates.sort(key=lambda x: len(x[1]))
        cta = cta_candidates[0][1]
        cta_index = cta_candidates[0][0]
        lines = [line for i, line in enumerate(lines) if i != cta_index]
    
    # Strategy 3: Second line is often subtext (if headline exists)
    if headline and lines:
        # Subtext is usually longer than headline but shorter than body
        if len(lines[0]) > len(headline) and len(lines[0]) < 200:
            subtext = lines[0]
            lines = lines[1:]
    
    # Remaining lines are body text
    body = '\n'.join(lines) if lines else None
    
    # Calculate confidence based on how well we identified structure
    confidence = 0.5  # Base confidence
    if headline:
        confidence += 0.2
    if cta:
        confidence += 0.2
    if subtext:
        confidence += 0.1
    
    return {
        'headline': headline,
        'subtext': subtext,
        'cta': cta,
        'body': body,
        'structure_confidence': min(confidence, 1.0),
        'all_text': text  # Keep original for reference
    }


def get_text_component_for_word(word: str, text_structure: Dict[str, Any]) -> Optional[str]:
    """
    Determine which text component (headline, subtext, CTA, body) contains a word.
    
    Args:
        word: Word to locate
        text_structure: Result from identify_text_structure()
        
    Returns:
        Component name ('headline', 'subtext', 'cta', 'body', or None)
    """
    word_lower = word.lower()
    
    for component in ['headline', 'subtext', 'cta', 'body']:
        component_text = text_structure.get(component)
        if component_text:
            # Check if word appears in this component
            component_words = re.findall(r'\b\w+\b', component_text.lower())
            if word_lower in component_words:
                return component
    
    return None

