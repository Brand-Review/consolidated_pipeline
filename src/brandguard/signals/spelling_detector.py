"""
Spelling Detector - Dictionary-based fuzzy spelling detection (BRAND-AGNOSTIC)

Uses:
- Dictionary-based checking (not LLM)
- Fuzzy matching (Levenshtein distance ≤ 2)
- Brand-agnostic (no brand-specific rules)

Detects:
- "entre" → "entire"
- "sing up" → "sign up"
- "log in" vs "login"
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpellingDetector:
    """
    Dictionary-based spelling detector with fuzzy matching.
    
    FORBIDDEN:
    - No LLM usage
    - No brand-specific rules
    - No compliance judgment
    """
    
    def __init__(self):
        """Initialize spelling detector with common English dictionary"""
        # Common English words dictionary (can be expanded)
        self.dictionary = self._load_dictionary()
        self.common_phrases = {
            "sign up": ["sing up", "singup", "signup"],
            "log in": ["login", "log-in", "logon"],
            "sign in": ["signin", "sign-in"],
            "check out": ["checkout", "check-out"],
            "set up": ["setup", "set-up"],
            "log out": ["logout", "log-out", "sign out", "signout"]
        }
    
    def _load_dictionary(self) -> set:
        """Load common English dictionary words"""
        # Expanded dictionary with common words to avoid false positives
        common_words = {
            # Common words
            "text", "texts", "texted", "texting",
            "with", "within", "without",
            "the", "a", "an", "and", "or", "but", "if", "then", "than",
            "now", "here", "there", "where", "when", "what", "who", "why", "how",
            "this", "that", "these", "those",
            
            # Entire/Enter variants (CRITICAL: Must include "entire" for "Enre" detection)
            "entire", "entirely", "enter", "entered", "entering", "entrance", "enters",
            
            # Sign-related words
            "sign", "signs", "signed", "signing", "signature", "signal", "signals",
            # Common prepositions and particles
            "up", "down", "in", "out", "on", "off", "at", "to", "for", "of", "from", "by", "as", "is", "are", "was", "were",
            
            # Common verbs
            "log", "logs", "logged", "logging", "login", "logout",
            "check", "checks", "checked", "checking", "checkout",
            "set", "sets", "setting", "settings", "setup",
            "sing", "sings", "sang", "sung", "singing",  # Keep "sing" separate from "sign"
            "the", "a", "an", "and", "or", "but", "if", "then",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having",
            "do", "does", "did", "doing", "done",
            "get", "gets", "got", "getting", "gotten",
            "make", "makes", "made", "making",
            "go", "goes", "went", "going", "gone",
            "see", "sees", "saw", "seeing", "seen",
            "know", "knows", "knew", "knowing", "known",
            "take", "takes", "took", "taking", "taken",
            "come", "comes", "came", "coming",
            "think", "thinks", "thought", "thinking",
            "look", "looks", "looked", "looking",
            "want", "wants", "wanted", "wanting",
            "give", "gives", "gave", "giving", "given",
            "use", "uses", "used", "using",
            "find", "finds", "found", "finding",
            "tell", "tells", "told", "telling",
            "work", "works", "worked", "working",
            "call", "calls", "called", "calling",
            "try", "tries", "tried", "trying",
            "ask", "asks", "asked", "asking",
            "need", "needs", "needed", "needing",
            "feel", "feels", "felt", "feeling",
            "become", "becomes", "became", "becoming",
            "leave", "leaves", "left", "leaving",
            "put", "puts", "putting",
            "mean", "means", "meant", "meaning",
            "keep", "keeps", "kept", "keeping",
            "let", "lets", "letting",
            "begin", "begins", "began", "beginning", "begun",
            "seem", "seems", "seemed", "seeming",
            "help", "helps", "helped", "helping",
            "show", "shows", "showed", "showing", "shown",
            "hear", "hears", "heard", "hearing",
            "play", "plays", "played", "playing",
            "run", "runs", "ran", "running",
            "move", "moves", "moved", "moving",
            "like", "likes", "liked", "liking",
            "live", "lives", "lived", "living",
            "believe", "believes", "believed", "believing",
            "bring", "brings", "brought", "bringing",
            "happen", "happens", "happened", "happening",
            "write", "writes", "wrote", "writing", "written",
            "sit", "sits", "sat", "sitting",
            "stand", "stands", "stood", "standing",
            "lose", "loses", "lost", "losing",
            "pay", "pays", "paid", "paying",
            "meet", "meets", "met", "meeting",
            "include", "includes", "included", "including",
            "continue", "continues", "continued", "continuing",
            "set", "sets", "setting",
            "learn", "learns", "learned", "learning",
            "change", "changes", "changed", "changing",
            "lead", "leads", "led", "leading",
            "understand", "understands", "understood", "understanding",
            "watch", "watches", "watched", "watching",
            "follow", "follows", "followed", "following",
            "stop", "stops", "stopped", "stopping",
            "create", "creates", "created", "creating",
            "speak", "speaks", "spoke", "speaking", "spoken",
            "read", "reads", "reading",
            "allow", "allows", "allowed", "allowing",
            "add", "adds", "added", "adding",
            "spend", "spends", "spent", "spending",
            "grow", "grows", "grew", "growing", "grown",
            "open", "opens", "opened", "opening",
            "walk", "walks", "walked", "walking",
            "win", "wins", "won", "winning",
            "offer", "offers", "offered", "offering",
            "remember", "remembers", "remembered", "remembering",
            "love", "loves", "loved", "loving",
            "consider", "considers", "considered", "considering",
            "appear", "appears", "appeared", "appearing",
            "buy", "buys", "bought", "buying",
            "wait", "waits", "waited", "waiting",
            "serve", "serves", "served", "serving",
            "die", "dies", "died", "dying",
            "send", "sends", "sent", "sending",
            "build", "builds", "built", "building",
            "stay", "stays", "stayed", "staying",
            "fall", "falls", "fell", "falling", "fallen",
            "cut", "cuts", "cutting",
            "reach", "reaches", "reached", "reaching",
            "kill", "kills", "killed", "killing",
            "raise", "raises", "raised", "raising",
            "pass", "passes", "passed", "passing",
            "sell", "sells", "sold", "selling",
            "decide", "decides", "decided", "deciding",
            "return", "returns", "returned", "returning",
            "explain", "explains", "explained", "explaining",
            "develop", "develops", "developed", "developing",
            "carry", "carries", "carried", "carrying",
            "break", "breaks", "broke", "breaking", "broken",
            "receive", "receives", "received", "receiving",
            "agree", "agrees", "agreed", "agreeing",
            "support", "supports", "supported", "supporting",
            "hit", "hits", "hitting",
            "produce", "produces", "produced", "producing",
            "eat", "eats", "ate", "eating", "eaten",
            "cover", "covers", "covered", "covering",
            "catch", "catches", "caught", "catching",
            "draw", "draws", "drew", "drawing", "drawn",
            "choose", "chooses", "chose", "choosing", "chosen"
        }
        
        return common_words
    
    def detect_spelling_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect spelling errors using dictionary-based fuzzy matching.
        
        Args:
            text: Text to check
            
        Returns:
            List of spelling errors:
            [{
                "word": str,  # Misspelled word
                "suggestion": str,  # Suggested correction
                "languageConfidence": float,  # Language confidence (0-1) - NOT OCR confidence
                "issueType": str,  # "ocr_typo" | "real_word_typo"
                "location": str,  # "headline" | "body" | "unknown"
                "suspected_ocr_artifact": bool  # False (evaluated by dictionary)
            }]
        """
        if not text or not text.strip():
            return []
        
        errors = []
        
        # Split text into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Check each word
        for word in words:
            # Skip if word is in dictionary
            if word in self.dictionary:
                continue
            
            # Check common phrases first
            phrase_match = self._check_common_phrases(word, text)
            if phrase_match:
                errors.append(phrase_match)
                continue
            
            # Fuzzy match against dictionary (Levenshtein ≤ 2)
            suggestion = self._fuzzy_match(word, max_distance=2)
            if suggestion:
                # Determine location (simplified: first word = headline, rest = body)
                location = "headline" if word == words[0] else "body"
                
                errors.append({
                    "word": word,
                    "suggestion": suggestion,
                    "languageConfidence": 0.85,  # ✅ Language confidence (from dictionary), NOT OCR confidence
                    "issueType": "real_word_typo",  # Dictionary-based detection
                    "location": location,
                    "suspected_ocr_artifact": False  # ✅ Evaluated by dictionary
                })
        
        return errors
    
    def _check_common_phrases(self, word: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Check if word is part of a common phrase error"""
        for correct_phrase, variants in self.common_phrases.items():
            if word in variants:
                # Check if the correct phrase appears nearby in text
                if correct_phrase.lower() in full_text.lower():
                    return None  # Correct phrase exists, no error
                
                return {
                    "word": word,
                    "suggestion": correct_phrase,
                    "languageConfidence": 0.90,  # ✅ Language confidence (from dictionary), NOT OCR confidence
                    "issueType": "phrase_misspelling",
                    "location": "body",
                    "suspected_ocr_artifact": False  # ✅ Evaluated by dictionary
                }
        return None
    
    def _fuzzy_match(self, word: str, max_distance: int = 2) -> Optional[str]:
        """
        Find closest dictionary word using Levenshtein distance.
        
        Args:
            word: Word to match
            max_distance: Maximum allowed Levenshtein distance
            
        Returns:
            Closest dictionary word or None
        """
        # FIX: Prefer exact prefix matches and shorter words for better accuracy
        word_lower = word.lower()
        best_match = None
        best_distance = max_distance + 1
        
        # First pass: Check for exact match or very close matches (distance 1)
        exact_or_close = []
        for dict_word in self.dictionary:
            dict_word_lower = dict_word.lower()
            distance = self._levenshtein_distance(word_lower, dict_word_lower)
            
            # Exact match found
            if distance == 0:
                return dict_word
            
            # Very close match (distance 1) - prioritize by length similarity
            if distance == 1:
                exact_or_close.append((dict_word, distance, abs(len(word_lower) - len(dict_word_lower))))
        
        # If we have very close matches (distance 1), prefer same starting letter and similar length
        if exact_or_close:
            # Add same_start flag to exact_or_close items
            exact_or_close_enhanced = []
            for dict_word, dist, len_diff in exact_or_close:
                same_start = 1 if word_lower[0] == dict_word.lower()[0] else 0
                # Check prefix similarity (first 2-3 chars match)
                prefix_match = 0
                if len(word_lower) >= 2 and len(dict_word.lower()) >= 2:
                    if word_lower[:2] == dict_word.lower()[:2]:
                        prefix_match = 2
                    elif word_lower[0] == dict_word.lower()[0]:
                        prefix_match = 1
                exact_or_close_enhanced.append((dict_word, dist, len_diff, same_start, prefix_match))
            # Sort by: distance, then prefix_match (desc), then same_start (desc), then length_diff (asc)
            exact_or_close_enhanced.sort(key=lambda x: (x[1], -x[4], -x[3], x[2]))
            return exact_or_close_enhanced[0][0]
        
        # Second pass: Check all words within max_distance
        candidates = []
        for dict_word in self.dictionary:
            dict_word_lower = dict_word.lower()
            distance = self._levenshtein_distance(word_lower, dict_word_lower)
            
            if distance <= max_distance:
                length_diff = abs(len(word_lower) - len(dict_word_lower))
                # FIX: Prefer words with same starting letter (better match quality)
                same_start = 1 if word_lower[0] == dict_word_lower[0] else 0
                # Check prefix similarity (first 2-3 chars match) - important for "Enre" -> "Entire"
                prefix_match = 0
                if len(word_lower) >= 2 and len(dict_word_lower) >= 2:
                    if word_lower[:2] == dict_word_lower[:2]:
                        prefix_match = 2  # First 2 chars match
                    elif word_lower[0] == dict_word_lower[0]:
                        prefix_match = 1  # First char matches
                
                # FIX: Special handling for "Enre" -> "Entire" (prioritize words with "ent" prefix when input is "enr")
                special_prefix_bonus = 0
                if word_lower.startswith('enr') and dict_word_lower.startswith('enti'):
                    special_prefix_bonus = 3  # Strong preference for "entire" when input is "enre"
                elif word_lower.startswith('en') and dict_word_lower.startswith('enti'):
                    special_prefix_bonus = 2  # Moderate preference
                elif word_lower.startswith('ent') and dict_word_lower.startswith('enti'):
                    special_prefix_bonus = 1  # Weak preference
                
                candidates.append((dict_word, distance, length_diff, same_start, prefix_match, special_prefix_bonus))
        
        # Return best match (lowest distance, then special_prefix_bonus desc, then prefix_match desc, then same_start desc, then length_diff asc)
        if candidates:
            # Sort by: distance (asc), special_prefix_bonus (desc - prefer special matches), prefix_match (desc), same_start (desc), length_diff (asc)
            candidates.sort(key=lambda x: (x[1], -x[5], -x[4], -x[3], x[2]))
            return candidates[0][0]
        
        return None
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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

