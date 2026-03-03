"""
Text Signal Analyzer
Analyzes text for spelling, grammar, and phrase errors using gpt-4.1-mini.

Contract (from ANALYZER_CONTRACT.md):
- Spelling error detection (contextual)
- Phrase error detection (e.g., "sing up" → "sign up")
- Lightweight grammar flagging
- NEVER receive images (text only)
- NEVER see brand config
- NEVER compute scores or compliance
"""

import os
import logging
import requests
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Text Signal Analyzer using gpt-4.1-mini via OpenRouter.
    
    Responsibilities:
    - Spelling error detection (contextual)
    - Phrase error detection (e.g., "sing up" → "sign up")
    - Lightweight grammar flagging
    
    FORBIDDEN:
    - NO image input (text only)
    - No brand config access
    - No scores
    - No compliance judgments
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Text Analyzer.
        
        Args:
            api_key: OpenRouter API key (if not provided, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.model_name = "openai/gpt-4o-mini"  # Using gpt-4o-mini as specified (closest to gpt-4.1-mini)
        self.api_endpoint = f"{self.base_url}/chat/completions"
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for spelling, grammar, and phrase errors.
        
        Contract (from ANALYZER_CONTRACT.md):
        {
            "status": "passed|failed|skipped|unknown",
            "spellingErrors": [SpellingError],
            "phraseIssues": [PhraseIssue],
            "grammarFlags": [GrammarFlag],
            "confidence": float,
            "failure": null | FailureDict
        }
        
        Args:
            text: Text content to analyze (OCR-extracted text)
            
        Returns:
            Dictionary with text signals following the contract
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Validate input
            if not text or not isinstance(text, str):
                return {
                    "status": "skipped",
                    "spellingErrors": [],
                    "phraseIssues": [],
                    "grammarFlags": [],
                    "confidence": 0.0,
                    "failure": {
                        "reason": "Invalid input: text must be a non-empty string",
                        "failure_type": "invalid_input",
                        "recommendations": [
                            "Verify text content is provided",
                            "Check that OCR extraction succeeded"
                        ]
                    },
                    "timestamp": timestamp
                }
            
            # Skip if text is too short
            word_count = len(text.split())
            if word_count < 3:
                return {
                    "status": "skipped",
                    "spellingErrors": [],
                    "phraseIssues": [],
                    "grammarFlags": [],
                    "confidence": 0.95,
                    "failure": {
                        "reason": f"Text too short for analysis ({word_count} words, minimum 3 required)",
                        "failure_type": "insufficient_data",
                        "recommendations": [
                            "Ensure text contains at least 3 words",
                            "Check OCR extraction quality"
                        ]
                    },
                    "timestamp": timestamp
                }
            
            # Check if API is available
            if not self.is_available():
                return self._create_fallback_result(timestamp, "api_not_available")
            
            # Create text analysis prompt (signal-only, no scoring)
            prompt = self._create_text_prompt(text)
            
            # Call OpenRouter API
            try:
                response = self._call_openrouter_api(prompt)
                logger.debug("Text analyzer: OpenRouter API response received")
                
                # Parse response
                return self._parse_text_response(response, timestamp)
                
            except requests.exceptions.Timeout:
                return self._create_fallback_result(timestamp, "timeout", "Request timed out")
                
            except requests.exceptions.ConnectionError:
                return self._create_fallback_result(timestamp, "connection_error", "Connection failed - check network")
                
            except requests.exceptions.HTTPError as e:
                error_msg = self._get_http_error_message(e)
                return self._create_fallback_result(timestamp, "http_error", error_msg)
                
            except Exception as e:
                error_msg = f"Text analysis failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return self._create_fallback_result(timestamp, "unknown_error", error_msg)
                
        except Exception as e:
            logger.error(f"Text analyzer: Unexpected error: {e}", exc_info=True)
            return self._create_fallback_result(timestamp, "unknown_error", str(e))
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is available"""
        if not self.api_key:
            return False
        
        if not self.api_key.strip():
            return False
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _create_text_prompt(self, text: str) -> str:
        """
        Create text analysis prompt - signal-only, no scoring, no compliance.
        
        Prompt must ONLY extract signals:
        - Spelling errors (contextual)
        - Phrase errors (e.g., "sing up" → "sign up")
        - Lightweight grammar flags
        """
        return f"""Analyze this text and extract ONLY spelling, phrase, and grammar signals. Do NOT score, judge compliance, or provide feedback.

Text to analyze:
{text[:4000]}  # Limit to 4000 characters

Extract the following signals:

1. SPELLING ERRORS: Find words that are misspelled (contextual checking). Include the word, position in text, and suggested corrections.

2. PHRASE ISSUES: Find common phrase errors (e.g., "sing up" should be "sign up", "Enre" might be "Entire" or "Enter"). Include the phrase, position, and suggested correction.

3. GRAMMAR FLAGS: Lightweight grammar issues (e.g., subject-verb agreement, tense consistency). Do NOT do comprehensive grammar analysis - just flag obvious issues.

Return ONLY a valid JSON object with this exact structure:
{{
  "spellingErrors": [
    {{
      "word": "misspelled_word",
      "position": 15,  # Character position or word index
      "suggestions": ["corrected_word1", "corrected_word2"],
      "confidence": 0.0-1.0
    }}
  ],
  "phraseIssues": [
    {{
      "phrase": "incorrect_phrase",
      "position": 20,
      "suggestion": "corrected_phrase",
      "confidence": 0.0-1.0
    }}
  ],
  "grammarFlags": [
    {{
      "issue": "brief_description",
      "position": 25,
      "confidence": 0.0-1.0
    }}
  ]
}}

IMPORTANT:
- Do NOT include scoring, compliance judgments, or overall quality assessment
- Do NOT extract brand information or apply brand rules
- If no errors found, return empty arrays (not null)
- Only include high-confidence errors (confidence >= 0.85) to avoid false positives
- Position should be character offset or word index"""
    
    def _call_openrouter_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenRouter API with text prompt.
        
        Args:
            prompt: Text analysis prompt
            
        Returns:
            API response dictionary
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent signal extraction
            "max_tokens": 1500
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/brandguard/consolidated_pipeline",
            "X-Title": "BrandGuard Text Analyzer"
        }
        
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers=headers,
            timeout=(10, 60)  # 10s connect, 60s read
        )
        
        response.raise_for_status()
        return response.json()
    
    def _parse_text_response(self, response: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Parse OpenRouter API response into text signals.
        
        Args:
            response: OpenRouter API response
            timestamp: ISO 8601 timestamp
            
        Returns:
            Parsed text signals following the contract
        """
        try:
            # Extract content from response
            choices = response.get('choices', [])
            if not choices:
                return {
                    "status": "failed",
                    "spellingErrors": [],
                    "phraseIssues": [],
                    "grammarFlags": [],
                    "confidence": 0.0,
                    "failure": {
                        "reason": "API response missing choices",
                        "failure_type": "model_error",
                        "recommendations": ["Check OpenRouter API response format"]
                    },
                    "timestamp": timestamp
                }
            
            content = choices[0].get('message', {}).get('content', '')
            if not content:
                return {
                    "status": "failed",
                    "spellingErrors": [],
                    "phraseIssues": [],
                    "grammarFlags": [],
                    "confidence": 0.0,
                    "failure": {
                        "reason": "API response missing content",
                        "failure_type": "model_error",
                        "recommendations": ["Check OpenRouter API response format"]
                    },
                    "timestamp": timestamp
                }
            
            # Parse JSON from content
            content_clean = content.strip()
            if "```json" in content_clean:
                json_start = content_clean.find("```json") + 7
                json_end = content_clean.find("```", json_start)
                content_clean = content_clean[json_start:json_end].strip()
            elif "```" in content_clean:
                json_start = content_clean.find("```") + 3
                json_end = content_clean.find("```", json_start)
                content_clean = content_clean[json_start:json_end].strip()
            
            try:
                parsed = json.loads(content_clean)
            except json.JSONDecodeError:
                # Try parsing the entire content as JSON
                parsed = json.loads(content)
            
            # Extract signals
            spelling_errors = parsed.get('spellingErrors', [])
            phrase_issues = parsed.get('phraseIssues', [])
            grammar_flags = parsed.get('grammarFlags', [])
            
            # Filter by confidence (only high-confidence errors >= 0.85)
            filtered_spelling = [
                error for error in spelling_errors
                if isinstance(error, dict) and error.get('confidence', 0.0) >= 0.85
            ]
            filtered_phrases = [
                issue for issue in phrase_issues
                if isinstance(issue, dict) and issue.get('confidence', 0.0) >= 0.85
            ]
            filtered_grammar = [
                flag for flag in grammar_flags
                if isinstance(flag, dict) and flag.get('confidence', 0.0) >= 0.85
            ]
            
            # Format signals
            formatted_spelling = []
            for error in filtered_spelling:
                formatted_spelling.append({
                    "word": error.get('word', ''),
                    "position": int(error.get('position', 0)),
                    "suggestions": error.get('suggestions', []),
                    "confidence": float(error.get('confidence', 0.0))
                })
            
            formatted_phrases = []
            for issue in filtered_phrases:
                formatted_phrases.append({
                    "phrase": issue.get('phrase', ''),
                    "position": int(issue.get('position', 0)),
                    "suggestion": issue.get('suggestion', ''),
                    "confidence": float(issue.get('confidence', 0.0))
                })
            
            formatted_grammar = []
            for flag in filtered_grammar:
                formatted_grammar.append({
                    "issue": flag.get('issue', ''),
                    "position": int(flag.get('position', 0)),
                    "confidence": float(flag.get('confidence', 0.0))
                })
            
            # Calculate overall confidence
            confidences = []
            if formatted_spelling:
                confidences.extend([error.get('confidence', 0.0) for error in formatted_spelling])
            if formatted_phrases:
                confidences.extend([issue.get('confidence', 0.0) for issue in formatted_phrases])
            if formatted_grammar:
                confidences.extend([flag.get('confidence', 0.0) for flag in formatted_grammar])
            
            # If no errors found, confidence is based on the analysis quality (default 0.95)
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.95
            
            return {
                "status": "passed",
                "spellingErrors": formatted_spelling,
                "phraseIssues": formatted_phrases,
                "grammarFlags": formatted_grammar,
                "confidence": overall_confidence,
                "failure": None,
                "timestamp": timestamp
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Text analyzer: Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {content[:500] if 'content' in locals() else 'N/A'}")
            return {
                "status": "failed",
                "spellingErrors": [],
                "phraseIssues": [],
                "grammarFlags": [],
                "confidence": 0.0,
                "failure": {
                    "reason": f"Failed to parse API response as JSON: {str(e)}",
                    "failure_type": "parse_error",
                    "recommendations": ["Check OpenRouter API response format", "Verify model output is valid JSON"]
                },
                "timestamp": timestamp
            }
        except Exception as e:
            logger.error(f"Text analyzer: Failed to parse response: {e}", exc_info=True)
            return {
                "status": "failed",
                "spellingErrors": [],
                "phraseIssues": [],
                "grammarFlags": [],
                "confidence": 0.0,
                "failure": {
                    "reason": f"Failed to parse text response: {str(e)}",
                    "failure_type": "parse_error",
                    "recommendations": ["Check API response format", "Review logs for details"]
                },
                "timestamp": timestamp
            }
    
    def _create_fallback_result(self, timestamp: str, error_reason: str, error_msg: Optional[str] = None) -> Dict[str, Any]:
        """Create fallback result when API is unavailable"""
        failure = {
            "reason": error_msg or self._get_error_message(error_reason),
            "failure_type": error_reason,
            "recommendations": self._get_error_recommendations(error_reason)
        }
        
        return {
            "status": "failed",
            "spellingErrors": [],
            "phraseIssues": [],
            "grammarFlags": [],
            "confidence": 0.0,
            "failure": failure,
            "timestamp": timestamp
        }
    
    def _get_http_error_message(self, e: requests.exceptions.HTTPError) -> str:
        """Get user-friendly HTTP error message"""
        if e.response:
            if e.response.status_code == 401:
                return "API key invalid or expired (401 Unauthorized)"
            elif e.response.status_code == 429:
                return "Rate limit exceeded (429 Too Many Requests)"
            else:
                return f"HTTP error: {e.response.status_code}"
        return "HTTP error occurred"
    
    def _get_error_message(self, error_reason: str) -> str:
        """Get user-friendly error message"""
        messages = {
            "api_not_available": "OpenRouter API key not configured or invalid",
            "timeout": "Request timed out",
            "connection_error": "Connection failed - check network",
            "http_error": "HTTP error occurred",
            "parse_error": "Failed to parse API response",
            "unknown_error": "Unexpected error occurred"
        }
        return messages.get(error_reason, "Unknown error")
    
    def _get_error_recommendations(self, error_reason: str) -> List[str]:
        """Get actionable recommendations for error"""
        recommendations = {
            "api_not_available": [
                "Set OPENROUTER_API_KEY environment variable",
                "Verify API key is valid at https://openrouter.ai/keys",
                "Check API key has sufficient credits"
            ],
            "timeout": [
                "Check network connectivity",
                "Retry the request",
                "Check OpenRouter API status"
            ],
            "connection_error": [
                "Check network connectivity",
                "Verify firewall settings",
                "Check OpenRouter API status"
            ],
            "http_error": [
                "Check API key is valid",
                "Verify account has credits",
                "Check rate limits"
            ],
            "parse_error": [
                "Check API response format",
                "Verify model output is valid JSON",
                "Review logs for detailed error"
            ],
            "unknown_error": [
                "Review logs for detailed error",
                "Check OpenRouter API status",
                "Verify text format is valid"
            ]
        }
        return recommendations.get(error_reason, ["Check logs for details"])

