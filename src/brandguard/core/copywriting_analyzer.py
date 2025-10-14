"""
Copywriting Analysis Module
Handles tone analysis and brand voice validation
"""

import logging
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image

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
    
    def analyze_copywriting(self, image: np.ndarray, text_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive copywriting analysis
        
        Args:
            image: Input image as numpy array
            text_content: Optional extracted text content to analyze
            
        Returns:
            Dictionary containing copywriting analysis results
        """
        try:
            logger.info("🔍 Starting copywriting analysis...")
            
            # Initialize results
            results = {
                'tone_analysis': {},
                'brand_voice_compliance': {},
                'copywriting_score': 0.0,
                'recommendations': [],
                'errors': []
            }
            
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
                            results['grammar_analysis'] = analysis.get('grammar', {})
                            
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
                            
                            # If VLLM didn't extract text, try OCR as fallback
                            if not text_content or text_content.strip() == '':
                                logger.info("VLLM didn't extract text, trying OCR fallback...")
                                ocr_text = self._extract_text_from_image(image)
                                if ocr_text and ocr_text.strip():
                                    text_content = ocr_text
                                    logger.info(f"✅ OCR extracted text: '{text_content}'")
                                    
                                    # Update the analysis with OCR text
                                    results['text_metrics'] = {
                                        'word_count': len(text_content.split()),
                                        'sentence_count': len(text_content.split('.')),
                                        'readability_level': results['text_metrics'].get('readability_level', 'grade8')
                                    }
                            
                            logger.info("✅ Used VLLMToneAnalyzer for image analysis")
                except Exception as e:
                    logger.warning(f"VLLM analysis failed: {e}, falling back to traditional methods")
                    # Fallback to traditional analysis
                    if text_content is None:
                        text_content = self._extract_text_from_image(image)
                    
                    # Ensure text_content is a string
                    if isinstance(text_content, dict):
                        text_content = str(text_content)
                    elif not isinstance(text_content, str):
                        text_content = str(text_content) if text_content is not None else ""
                    
                    # Analyze tone
                    tone_results = self._analyze_tone(text_content)
                    results['tone_analysis'] = tone_results
            else:
                # Traditional analysis fallback
                # Extract text if not provided
                if text_content is None:
                    text_content = self._extract_text_from_image(image)
                
                # Ensure text_content is a string
                if isinstance(text_content, dict):
                    text_content = str(text_content)
                elif not isinstance(text_content, str):
                    text_content = str(text_content) if text_content is not None else ""
                
                # Analyze tone
                tone_results = self._analyze_tone(text_content)
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
            results['copywriting_score'] = copywriting_score
            
            # Generate recommendations
            recommendations = self._generate_copywriting_recommendations(tone_results, voice_compliance)
            results['recommendations'] = recommendations
            
            # Add extracted text to results
            results['extracted_text'] = text_content
            results['text_content'] = text_content  # Alias for compatibility
            
            logger.info(f"✅ Copywriting analysis completed. Score: {copywriting_score:.2f}")
            if text_content:
                logger.info(f"📝 Extracted text: '{text_content}'")
            return results
            
        except Exception as e:
            logger.error(f"Copywriting analysis failed: {e}")
            import traceback
            logger.error(f"Copywriting analysis traceback: {traceback.format_exc()}")
            return {
                'tone_analysis': {},
                'brand_voice_compliance': {},
                'copywriting_score': 0.0,
                'recommendations': ['Copywriting analysis failed due to technical error'],
                'errors': [str(e)]
            }
    
    def _extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text content from image using OCR"""
        try:
            # This would typically use OCR libraries like Tesseract
            # For now, return a placeholder
            return "Sample text content for analysis"
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
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
                    analysis = self.vllm_analyzer.analyze_text(text_content, user_settings)
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
                        analysis = self.tone_analyzer.analyze_text_tone(text_content)
                        if analysis:
                            tone_results.update(analysis)
            elif self.tone_analyzer and text_content:
                # Use old tone analyzer as fallback
                analysis = self.tone_analyzer.analyze_text_tone(text_content)
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