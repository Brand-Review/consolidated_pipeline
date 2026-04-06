"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description:Base Pipeline Orchestrator 
Core functionality for coordinating all BrandGuard models
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .model_imports import import_all_models, get_imported_models, is_models_loaded
from .color_analyzer import ColorAnalyzer
from .logo_analyzer import LogoAnalyzer
from .typography_analyzer import TypographyAnalyzer
from .copywriting_analyzer import CopywritingAnalyzer
from .brand_compliance_judge import BrandComplianceJudge
from ..brand_profile.brand_store import BrandStore
from ..brand_profile.text_rag import TextRAG
from ..brand_profile.asset_rag import AssetRAG

logger = logging.getLogger(__name__)

class BasePipelineOrchestrator:
    """
    Base orchestrator that coordinates all BrandGuard models
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.analysis_results = {}
        self.current_analysis_id = None

        # Brand profile services (shared, lazy)
        self.brand_store = BrandStore()
        self.text_rag = TextRAG()
        self.asset_rag = AssetRAG()

        # Import and initialize models
        self.MODELS_LOADED = import_all_models()
        self.imported_models = get_imported_models()

        # Initialize analyzers
        self._initialize_analyzers()

        # Unified LLM judge (lazy — uses OPENROUTER_API_KEY env var)
        self.brand_judge = BrandComplianceJudge()
    
    def _initialize_analyzers(self):
        """Initialize all analyzer components"""
        try:
            # Initialize color analyzer
            self.color_analyzer = ColorAnalyzer(self.settings, self.imported_models)
            
            # Initialize logo analyzer
            self.logo_analyzer = LogoAnalyzer(self.settings, self.imported_models)

            # Initialize typography analyzer
            self.typography_analyzer = TypographyAnalyzer(self.settings, self.imported_models)

            # Initialize copywriting analyzer
            self.copywriting_analyzer = CopywritingAnalyzer(self.settings, self.imported_models)
            
            logger.info("All analyzers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            import traceback
            logger.error(f"Analyzer initialization traceback: {traceback.format_exc()}")
    
    def analyze_content(self,
                       input_source: str,
                       source_type: str = 'image',
                       analysis_options: Optional[Dict[str, Any]] = None,
                       brand_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis method that coordinates all models.

        Args:
            input_source: Path to file, text content, or URL
            source_type: Type of input ('image', 'document', 'text', 'url')
            analysis_options: Configuration options for analysis
            brand_id: Optional brand profile ID. When provided, rules and RAG
                      context from the stored brand profile are injected into
                      each analyzer instead of the global YAML defaults.

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Generate analysis ID
            self.current_analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Load brand profile if provided
            brand_profile = None
            if brand_id:
                brand_profile = self.brand_store.get(brand_id)
                if not brand_profile:
                    logger.warning(f"Brand profile {brand_id} not found — falling back to global settings")

            # Initialize results structure
            results = {
                'analysis_id': self.current_analysis_id,
                'timestamp': datetime.now().isoformat(),
                'input_source': input_source,
                'source_type': source_type,
                'brand_id': brand_id,
                'model_results': {},
                'overall_compliance': 0.0,
                'summary': '',
                'recommendations': [],
                # Stashed for _calculate_overall_compliance; not sent to caller
                '_analysis_options': analysis_options or {},
            }

            # Route to appropriate analysis method
            if source_type == 'image':
                analysis_result = self._analyze_image(input_source, analysis_options, brand_id, brand_profile)
            elif source_type == 'document':
                analysis_result = self._analyze_document(input_source, analysis_options)
            elif source_type == 'text':
                analysis_result = self._analyze_text(input_source, analysis_options)
            elif source_type == 'url':
                analysis_result = self._analyze_url(input_source, analysis_options)
            else:
                return {'error': f'Unsupported source type: {source_type}'}
            
            # Merge results
            results.update(analysis_result)
            
            # Calculate overall compliance (also populates results['compliance_breakdown'])
            results['overall_compliance'] = self._calculate_overall_compliance(results)

            # Generate verdict (LLM verdict wins in 'llm' mode)
            pass_threshold = results.get('_pass_threshold', 0.70)
            judge_verdict = results.get('_judge_verdict')
            verdict_mode_final = results.pop('_verdict_mode', 'threshold')
            results['verdict'] = self._generate_verdict(
                results['overall_compliance'],
                pass_threshold,
                judge_verdict=judge_verdict,
                verdict_mode=verdict_mode_final,
                compliance_breakdown=results.get('compliance_breakdown'),
            )

            # Include verdict_reason when in llm mode and LLM provided one
            if verdict_mode_final == 'llm' and judge_verdict:
                results['verdict_reason'] = judge_verdict.get('verdict_reason', '')

            # Generate summary and recommendations
            summary_data = self._generate_summary_and_recommendations(results)
            results.update(summary_data)

            # Strip internal keys before returning
            for _key in ('_analysis_options', '_judge_verdict', '_pass_threshold'):
                results.pop(_key, None)

            # Store results
            self.analysis_results[self.current_analysis_id] = results

            return results
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_image(
        self,
        image_path: str,
        analysis_options: Optional[Dict[str, Any]] = None,
        brand_id: Optional[str] = None,
        brand_profile: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze image content.

        When brand_id is present: YOLO + k-means run first for fast objective data,
        then BrandComplianceJudge makes ONE OpenRouter call with full context
        (brand guidelines, YOLO bboxes, dominant colors, few-shot examples) and
        returns scores for all 4 dimensions plus extracted text, fonts, and verified
        logo bboxes (stored for YOLO fine-tuning).

        When brand_id is absent: falls back to rule-based analyzers for all 4
        dimensions (previously only color + logo were scored).
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}

            model_results: Dict[str, Any] = {}

            # ------------------------------------------------------------------
            # Step 1 — Always run YOLO (fast, objective bboxes)
            # ------------------------------------------------------------------
            logo_options = (analysis_options or {}).get('logo_analysis', {})
            if brand_profile:
                logo_rules = brand_profile.get("rules", {}).get("logo_rules", {})
                if logo_rules.get("allowed_zones") and not logo_options.get("allowed_zones"):
                    logo_options = dict(logo_options)
                    logo_options["allowed_zones"] = logo_rules["allowed_zones"]
                if logo_rules.get("min_height_px") and not logo_options.get("min_logo_size"):
                    logo_options = dict(logo_options)
                    logo_options["min_logo_size"] = logo_rules["min_height_px"]

            yolo_logo_result = self.logo_analyzer.analyze_logos(
                image.copy(), logo_options,
                rag_context="",
                few_shot_examples=[],
            )
            self._log_logo_analysis_terminal(yolo_logo_result)

            # Normalise YOLO bboxes for the LLM judge
            yolo_detections: List[Dict] = []
            for i, det in enumerate(yolo_logo_result.get("logo_detections") or []):
                bbox = det.get("bbox") or det.get("bounding_box") or []
                yolo_detections.append({
                    "detection_id": i,
                    "bbox": bbox,
                    "confidence": det.get("confidence", 0),
                    "label": det.get("label") or det.get("class_name", "logo"),
                })

            # ------------------------------------------------------------------
            # Step 2 — Always run k-means color extraction (objective pixel data)
            # ------------------------------------------------------------------
            color_options = dict((analysis_options or {}).get('color_analysis', {}))
            if brand_profile:
                color_rules = brand_profile.get("rules", {}).get("color_rules", {})
                if color_rules.get("palette") and not color_options.get("primary_colors"):
                    color_options["primary_colors"] = ",".join(color_rules["palette"])

            color_result = self.color_analyzer.analyze_colors(image, color_options)
            self._log_color_analysis_terminal(color_result)
            dominant_colors: List[Dict] = color_result.get("dominant_colors") or []

            # ------------------------------------------------------------------
            # Step 3 — BrandComplianceJudge (when brand_id present)
            # ------------------------------------------------------------------
            if brand_id:
                # Retrieve RAG context
                brand_context_parts: List[str] = []
                few_shot_examples: List[Dict] = []

                for check_type in ("color", "typography", "logo", "copywriting"):
                    try:
                        chunk = self.text_rag.retrieve(brand_id, check_type, top_k=3)
                        if chunk:
                            brand_context_parts.append(f"### {check_type.title()} Guidelines\n{chunk}")
                    except Exception as e:
                        logger.warning(f"RAG retrieval failed for {check_type}: {e}")

                try:
                    few_shot_examples = self.asset_rag.retrieve_similar(brand_id, image, top_k=3)
                except Exception as e:
                    logger.warning(f"Asset RAG retrieval failed: {e}")

                brand_context = "\n\n".join(brand_context_parts)
                verdict_mode = (analysis_options or {}).get('verdict_mode', 'threshold')

                # Extract structured brand rules from the stored profile
                brand_rules = brand_profile.get("rules", {}) if brand_profile else {}

                judge_verdict = self.brand_judge.run(
                    image_path=image_path,
                    brand_context=brand_context,
                    brand_rules=brand_rules,
                    dominant_colors=dominant_colors,
                    logo_detections=yolo_detections,
                    few_shot_examples=few_shot_examples,
                    verdict_mode=verdict_mode,
                )

                if judge_verdict:
                    # Merge YOLO detections with LLM logo judgements
                    logo_judgements = judge_verdict.get("logo_judgements") or []

                    # Build final logo_analysis from judge output
                    actual_logos = [j for j in logo_judgements if j.get("is_logo")]
                    logo_score = (
                        len(actual_logos) / max(1, len(logo_judgements))
                        if logo_judgements else 0.0
                    )

                    judge_logo_score = judge_verdict.get("logo", {}).get("score")
                    model_results['logo_analysis'] = {
                        **yolo_logo_result,
                        "logo_judgements": logo_judgements,
                        "scores": {
                            **yolo_logo_result.get("scores", {}),
                            "overall": round(judge_logo_score if judge_logo_score is not None else logo_score, 3),
                        },
                        "judge_scores": {
                            "score": judge_verdict.get("logo", {}).get("score", logo_score),
                            "reason": judge_verdict.get("logo", {}).get("reason", ""),
                        },
                    }

                    # Color analysis enriched with LLM verification
                    color_verification = judge_verdict.get("color_verification", {})
                    model_results['color_analysis'] = {
                        **color_result,
                        "color_verification": color_verification,
                        "brand_validation": {
                            **color_result.get("brand_validation", {}),
                            "compliance_score": judge_verdict.get("color", {}).get("score", 0.0),
                            "reason": judge_verdict.get("color", {}).get("reason", ""),
                        },
                    }

                    # Typography analysis from LLM (replaces PaddleOCR + font classifier)
                    model_results['typography_analysis'] = {
                        "extracted_text": judge_verdict.get("extracted_text", ""),
                        "fonts_detected": judge_verdict.get("detected_fonts", []),
                        "typography_score": judge_verdict.get("typography", {}).get("score", 0.0),
                        "reason": judge_verdict.get("typography", {}).get("reason", ""),
                        "source": "llm_judge",
                    }

                    # Copywriting analysis from LLM (replaces HybridToneAnalyzer)
                    extracted = judge_verdict.get("extracted_text", "")
                    words = extracted.split() if extracted else []
                    sentences = [s for s in extracted.replace('\n', '. ').split('.') if s.strip()] if extracted else []
                    model_results['copywriting_analysis'] = {
                        "extracted_text": extracted,
                        "copywriting_score": judge_verdict.get("copywriting", {}).get("score", 0.0),
                        "reason": judge_verdict.get("copywriting", {}).get("reason", ""),
                        "source": "llm_judge",
                        "text_metrics": {
                            "word_count": len(words),
                            "sentence_count": len(sentences),
                            "readability_level": "N/A",
                        },
                        "grammar_analysis": judge_verdict.get("grammar_analysis", {
                            "errors": [], "error_count": 0, "summary": "No grammar analysis available"
                        }),
                        "tone_analysis": judge_verdict.get("tone_analysis", {}),
                    }

                    return {
                        'model_results': model_results,
                        'analysis_type': 'image_analysis',
                        '_judge_verdict': judge_verdict,   # kept for _calculate_overall_compliance
                        '_verdict_mode': verdict_mode,     # kept for _generate_verdict
                    }
                else:
                    logger.warning("[_analyze_image] LLM judge returned None — falling back to rule-based")

            # ------------------------------------------------------------------
            # Step 4 — Rule-based fallback (no brand_id, or LLM judge failed)
            # ------------------------------------------------------------------
            model_results['color_analysis'] = color_result
            model_results['logo_analysis'] = yolo_logo_result

            # Typography fallback
            if (analysis_options or {}).get('typography_analysis', {}).get('enabled', True):
                model_results['typography_analysis'] = self.typography_analyzer.analyze_typography(
                    image, None,
                    rag_context="",
                    few_shot_examples=[],
                    brand_rules=brand_profile.get("rules", {}).get("typography_rules", {}) if brand_profile else None,
                )

            # Copywriting fallback
            if (analysis_options or {}).get('copywriting_analysis', {}).get('enabled', True):
                copywriting_options = (analysis_options or {}).get('copywriting_analysis', {})
                model_results['copywriting_analysis'] = self.copywriting_analyzer.analyze_copywriting(
                    image, copywriting_options,
                    rag_context="",
                    few_shot_examples=[],
                )

            return {
                'model_results': model_results,
                'analysis_type': 'image_analysis',
            }

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {'error': f'Image analysis failed: {str(e)}'}

    def _log_color_analysis_terminal(self, color_result: Dict[str, Any]) -> None:
        """Emit dominant colors in a readable form (matches historical server log style)."""
        try:
            rows = color_result.get('dominant_colors') or []
            if not rows:
                msg = "Color results: []"
                print(msg, flush=True)
                logger.info(msg)
                return
            out: List[Dict[str, Any]] = []
            for c in rows:
                rgb = c.get("rgb")
                if rgb is not None and hasattr(rgb, "__iter__") and not isinstance(rgb, (str, bytes)):
                    rgb = tuple(int(x) for x in rgb)
                pct = c.get("percentage", 0)
                if hasattr(pct, "item"):
                    pct = float(pct.item())
                else:
                    pct = float(pct)
                cid = c.get("cluster_id")
                out.append({
                    "rgb": rgb,
                    "hex": c.get("hex"),
                    "percentage": round(pct, 2),
                    "cluster_id": int(cid) if cid is not None else None,
                })
            msg = f"Color results: {out}"
            print(msg, flush=True)
            logger.info("Color results: %s", out)
        except Exception as e:
            logger.debug("Color terminal log skipped: %s", e)

    def _log_logo_analysis_terminal(self, logo_result: Dict[str, Any]) -> None:
        """Summarize logo pass for server logs (YOLO step lines still come from LogoDetector)."""
        try:
            dets = logo_result.get("logo_detections") or []
            n = len(dets) if isinstance(dets, list) else 0
            path = logo_result.get("pipeline_path")
            src = logo_result.get("detection_source")
            or_ok = logo_result.get("openrouter_available", True)
            if path or src is not None:
                msg = (
                    f"Logo analysis summary: {n} detection(s), "
                    f"pipeline_path={path!r}, detection_source={src!r}, "
                    f"openrouter_available={or_ok}"
                )
                print(msg, flush=True)
                logger.info(
                    "Logo analysis summary: %d detection(s), pipeline_path=%r, "
                    "detection_source=%r, openrouter_available=%s",
                    n,
                    path,
                    src,
                    or_ok,
                )
            else:
                msg = f"Logo analysis summary: {n} detection(s)"
                print(msg, flush=True)
                logger.info("Logo analysis summary: %d detection(s)", n)
        except Exception as e:
            logger.debug("Logo terminal log skipped: %s", e)

    def _analyze_document(self, document_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            # TODO: Implement document analysis
            return {
                'model_results': {},
                'analysis_type': 'document_analysis',
                'message': 'Document analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {'error': f'Document analysis failed: {str(e)}'}
    
    def _analyze_text(self, text_content: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze text content"""
        try:
            # TODO: Implement text analysis
            return {
                'model_results': {},
                'analysis_type': 'text_analysis',
                'message': 'Text analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {'error': f'Text analysis failed: {str(e)}'}
    
    def _analyze_url(self, url: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze URL content"""
        try:
            # TODO: Implement URL analysis
            return {
                'model_results': {},
                'analysis_type': 'url_analysis',
                'message': 'URL analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return {'error': f'URL analysis failed: {str(e)}'}
    
    def _calculate_overall_compliance(self, results: Dict[str, Any]) -> float:
        """
        Weighted compliance score across all 4 dimensions.

        Weights and pass_threshold come from analysis_options (sent by br-be).
        Defaults: 25% each, 70% pass threshold.
        """
        try:
            model_results = results.get('model_results', {})
            analysis_options = results.get('_analysis_options', {})
            judge_verdict = results.get('_judge_verdict')

            # Weights from br-be Config (serialised as JSON string by case-converter)
            raw_weights = analysis_options.get('scoring_weights') or analysis_options.get('scoring_weights', {})
            if isinstance(raw_weights, str):
                try:
                    raw_weights = json.loads(raw_weights)
                except Exception:
                    raw_weights = {}

            default_weight = 0.25
            weights = {
                'color':       float(raw_weights.get('color', default_weight)),
                'logo':        float(raw_weights.get('logo', default_weight)),
                'typography':  float(raw_weights.get('typography', default_weight)),
                'copywriting': float(raw_weights.get('copywriting', default_weight)),
            }

            # Extract per-dimension scores
            def _color_score() -> float:
                if judge_verdict:
                    return float(judge_verdict.get('color', {}).get('score', 0))
                ca = model_results.get('color_analysis', {})
                return float(ca.get('brand_validation', {}).get('compliance_score', 0))

            def _logo_score() -> float:
                if judge_verdict:
                    return float(judge_verdict.get('logo', {}).get('score', 0))
                la = model_results.get('logo_analysis', {})
                return float(la.get('scores', {}).get('overall', 0))

            def _typography_score() -> float:
                if judge_verdict:
                    return float(judge_verdict.get('typography', {}).get('score', 0))
                ta = model_results.get('typography_analysis', {})
                return float(
                    ta.get('typography_score')
                    or ta.get('scores', {}).get('overall', 0)
                    or 0
                )

            def _copywriting_score() -> float:
                if judge_verdict:
                    return float(judge_verdict.get('copywriting', {}).get('score', 0))
                cwa = model_results.get('copywriting_analysis', {})
                return float(
                    cwa.get('copywriting_score')
                    or cwa.get('scores', {}).get('overall', 0)
                    or 0
                )

            dimension_scores = {
                'color':       _color_score(),
                'logo':        _logo_score(),
                'typography':  _typography_score(),
                'copywriting': _copywriting_score(),
            }

            pass_threshold = float(analysis_options.get('pass_threshold', 0.70))

            # Build compliance_breakdown
            compliance_breakdown: Dict[str, Any] = {}
            for dim, score in dimension_scores.items():
                w = weights[dim]
                reason = ""
                if judge_verdict:
                    reason = judge_verdict.get(dim, {}).get('reason', '')
                compliance_breakdown[dim] = {
                    'score': round(score, 3),
                    'reason': reason,
                    'weight': round(w, 3),
                    'passed': score >= pass_threshold,
                }

            # Weighted sum
            total_weight = sum(weights.values()) or 1.0
            overall = sum(
                dimension_scores[dim] * weights[dim]
                for dim in dimension_scores
            ) / total_weight

            # Store for the response builder
            results['compliance_breakdown'] = compliance_breakdown
            results['_pass_threshold'] = pass_threshold

            return round(overall, 3)

        except Exception as e:
            logger.error(f"Compliance calculation failed: {e}")
            return 0.0

    # Any single dimension below this score forces rejection regardless of overall average.
    _CRITICAL_DIMENSION_THRESHOLD = 0.65

    def _generate_verdict(
        self,
        score: float,
        threshold: float,
        judge_verdict: Optional[Dict] = None,
        verdict_mode: str = 'threshold',
        compliance_breakdown: Optional[Dict] = None,
    ) -> str:
        """
        Return 'approved' or 'rejected'.

        In 'llm' mode: LLM explicit verdict always wins when present.
        In 'threshold' mode (default): overall score >= threshold, PLUS no single
        dimension may fall below _CRITICAL_DIMENSION_THRESHOLD (hard floor).
        """
        if (
            verdict_mode == 'llm'
            and judge_verdict
            and judge_verdict.get('verdict') in ('approved', 'rejected')
        ):
            return judge_verdict['verdict']

        # Hard-floor: any critical dimension failure overrides the overall average
        if compliance_breakdown:
            critical_fails = [
                dim for dim, data in compliance_breakdown.items()
                if isinstance(data, dict) and data.get('score', 1.0) < self._CRITICAL_DIMENSION_THRESHOLD
            ]
            if critical_fails:
                logger.info(
                    "[_generate_verdict] Hard-floor rejection — critical failures: %s",
                    ", ".join(critical_fails),
                )
                return 'rejected'

        return 'approved' if score >= threshold else 'rejected'
    
    def _generate_summary_and_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary and recommendations"""
        try:
            model_results = results.get('model_results', {})
            overall_compliance = results.get('overall_compliance', 0)
            
            # Generate summary (scores are 0.0–1.0)
            if overall_compliance >= 0.80:
                summary = "Excellent brand compliance"
            elif overall_compliance >= 0.60:
                summary = "Good brand compliance with minor issues"
            elif overall_compliance >= 0.40:
                summary = "Moderate brand compliance with several issues"
            else:
                summary = "Poor brand compliance requiring attention"
            
            # Generate recommendations
            recommendations = []
            
            # Color recommendations
            if 'color_analysis' in model_results:
                color_validation = model_results['color_analysis'].get('brand_validation', {})
                if not color_validation.get('valid', True):
                    recommendations.append("Review color palette to better align with brand guidelines")
            
            # Logo recommendations
            if 'logo_analysis' in model_results:
                logo_validation = model_results['logo_analysis'].get('placement_validation', {})
                if not logo_validation.get('valid', True):
                    recommendations.append("Adjust logo placement and sizing for better compliance")
            
            return {
                'summary': summary,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                'summary': 'Analysis completed with errors',
                'recommendations': ['Review analysis results for details']
            }
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get status of a specific analysis"""
        try:
            if analysis_id in self.analysis_results:
                return {
                    'analysis_id': analysis_id,
                    'status': 'completed',
                    'results': self.analysis_results[analysis_id]
                }
            else:
                return {'error': 'Analysis not found'}
                
        except Exception as e:
            logger.error(f"Failed to get analysis status: {e}")
            return {'error': f'Failed to get analysis status: {str(e)}'}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear analysis results
            self.analysis_results.clear()
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
