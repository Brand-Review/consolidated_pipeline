"""
Consolidated configuration settings for BrandGuard Pipeline
Combines settings from all four models: Color, Typography, Copywriting, and Logo Detection
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class BrandColor:
    """Brand color definition"""
    name: str
    hex_code: str
    rgb: tuple
    tolerance: float = 0.1

@dataclass
class BrandColorPalette:
    """Brand color palette configuration"""
    name: str
    primary_colors: List[BrandColor]
    secondary_colors: List[BrandColor] = field(default_factory=list)
    accent_colors: List[BrandColor] = field(default_factory=list)
    forbidden_colors: List[BrandColor] = field(default_factory=list)

@dataclass
class TypographyRules:
    """Typography and font compliance rules"""
    approved_fonts: List[str] = field(default_factory=list)
    max_font_size: int = 72
    min_font_size: int = 8
    preferred_font_families: List[str] = field(default_factory=list)
    forbidden_fonts: List[str] = field(default_factory=list)
    line_height_ratio: float = 1.5
    letter_spacing: float = 0.0

@dataclass
class BrandVoiceSettings:
    """Brand voice and tone preferences"""
    formality_score: int = 50  # 0-100
    confidence_level: str = "balanced"  # conservative, balanced, aggressive
    warmth_score: int = 50  # 0-100
    energy_score: int = 50  # 0-100
    readability_level: str = "grade8"  # grade6, grade8, grade10, grade12
    persona_type: str = "general"  # general, professional, casual, friendly
    allow_emojis: bool = False
    allow_slang: bool = False
    no_financial_guarantees: bool = True
    no_medical_claims: bool = True
    no_competitor_bashing: bool = True

@dataclass
class BrandRules:
    """Brand compliance rules for logo placement"""
    allowed_zones: List[str] = field(default_factory=lambda: ["top-left", "top-right", "bottom-left", "bottom-right"])
    min_logo_size: float = 0.01
    max_logo_size: float = 0.25
    min_edge_distance: float = 0.05
    aspect_ratio_tolerance: float = 0.2

@dataclass
class LogoDetectionSettings:
    """Logo detection and placement validation settings"""
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    min_logo_size: int = 20
    max_logo_size: int = 500
    placement_rules: Dict[str, Any] = field(default_factory=dict)
    
    # YOLOv8 nano configuration
    yolo_model: str = "yolov8n.pt"
    use_yolo: bool = True
    
    # Qwen2.5-VL-3B-Instruct configuration via vLLM
    use_qwen: bool = True
    qwen_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_api_url: str = "http://localhost:8000/v1/chat/completions"
    qwen_timeout: int = 120
    
    # Detection combination
    combine_detections: bool = True
    enhance_yolo_with_qwen: bool = True
    
    # LLVa with Ollama integration (legacy)
    enable_llva_ollama: bool = False
    llva_analysis_focus: str = "comprehensive"  # comprehensive, logo_only, context_only
    llva_model: str = "llava-1.5"
    llva_confidence_threshold: float = 0.7

@dataclass
class AnalysisSettings:
    """Analysis configuration for different models"""
    color_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'n_colors': 8,
        'n_clusters': 8,
        'color_tolerance': 0.2,
        'enable_contrast_check': True
    })
    
    typography_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'merge_regions': True,
        'distance_threshold': 20,
        'confidence_threshold': 0.7,
        'enable_font_validation': True
    })
    
    copywriting_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'include_suggestions': True,
        'include_industry_benchmarks': True,
        'enable_brand_profile_matching': True
    })
    
    logo_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'enable_placement_validation': True,
        'enable_brand_compliance': True,
        'generate_annotations': True
    })

@dataclass
class Settings:
    """Consolidated application settings"""
    
    # Model-specific settings
    color_palette: BrandColorPalette = field(default_factory=lambda: BrandColorPalette("default", []))
    typography_rules: TypographyRules = field(default_factory=TypographyRules)
    brand_voice: BrandVoiceSettings = field(default_factory=BrandVoiceSettings)
    logo_detection: LogoDetectionSettings = field(default_factory=LogoDetectionSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    
    # File paths and directories
    config_dir: str = "configs"
    upload_dir: str = "uploads"
    results_dir: str = "results"
    models_dir: str = "models"
    
    # API settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'pdf'
    ])
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_analyses: int = 5
    
    def __post_init__(self):
        """Initialize settings after creation"""
        self._load_configurations()
        self._create_directories()
    
    def _load_configurations(self):
        """Load configurations from files"""
        config_path = Path(self.config_dir)
        
        if not config_path.exists():
            self._create_default_configs()
            return
        
        # Load color palette config
        color_config = config_path / "color_palette.yaml"
        if color_config.exists():
            self._load_color_config(color_config)
        
        # Load typography config
        typography_config = config_path / "typography_rules.yaml"
        if typography_config.exists():
            self._load_typography_config(typography_config)
        
        # Load brand voice config
        voice_config = config_path / "brand_voice.yaml"
        if voice_config.exists():
            self._load_brand_voice_config(voice_config)
        
        # Load logo detection config
        logo_config = config_path / "logo_detection.yaml"
        if logo_config.exists():
            self._load_logo_config(logo_config)
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.config_dir, self.upload_dir, self.results_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _create_default_configs(self):
        """Create default configurations"""
        self._create_default_color_palette()
        self._create_default_typography_rules()
        self._create_default_brand_voice()
        self._create_default_logo_detection()
    
    def _create_default_color_palette(self):
        """Create default color palette"""
        self.color_palette = BrandColorPalette(
            name="Default",
            primary_colors=[
                BrandColor("Primary Blue", "#1E40AF", (30, 64, 175), 0.1),
                BrandColor("Primary Dark", "#1E293B", (30, 41, 59), 0.1),
            ],
            secondary_colors=[
                BrandColor("Secondary Gray", "#64748B", (100, 116, 139), 0.1),
                BrandColor("Light Gray", "#F1F5F9", (241, 245, 249), 0.1),
            ],
            accent_colors=[
                BrandColor("Accent Orange", "#F97316", (249, 115, 22), 0.1),
                BrandColor("Accent Green", "#10B981", (16, 185, 129), 0.1),
            ]
        )
    
    def _create_default_typography_rules(self):
        """Create default typography rules"""
        self.typography_rules = TypographyRules(
            approved_fonts=["Arial", "Helvetica", "Times New Roman", "Georgia"],
            preferred_font_families=["sans-serif", "serif"],
            max_font_size=72,
            min_font_size=8
        )
    
    def _create_default_brand_voice(self):
        """Create default brand voice settings"""
        self.brand_voice = BrandVoiceSettings(
            formality_score=60,
            confidence_level="balanced",
            warmth_score=50,
            energy_score=50
        )
    
    def _create_default_logo_detection(self):
        """Create default logo detection settings"""
        self.logo_detection = LogoDetectionSettings(
            confidence_threshold=0.5,
            iou_threshold=0.45,
            max_detections=100
        )
    
    def _load_color_config(self, config_path: Path):
        """Load color palette configuration"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'brand_palette' in config_data:
                palette_data = config_data['brand_palette']
                self.color_palette = self._create_palette_from_config(palette_data)
        except Exception as e:
            print(f"Error loading color config: {e}")
    
    def _load_typography_config(self, config_path: Path):
        """Load typography configuration"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'typography_rules' in config_data:
                rules_data = config_data['typography_rules']
                self.typography_rules = TypographyRules(**rules_data)
        except Exception as e:
            print(f"Error loading typography config: {e}")
    
    def _load_brand_voice_config(self, config_path: Path):
        """Load brand voice configuration"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'brand_voice' in config_data:
                voice_data = config_data['brand_voice']
                self.brand_voice = BrandVoiceSettings(**voice_data)
        except Exception as e:
            print(f"Error loading brand voice config: {e}")
    
    def _load_logo_config(self, config_path: Path):
        """Load logo detection configuration"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'logo_detection' in config_data:
                logo_data = config_data['logo_detection']
                self.logo_detection = LogoDetectionSettings(**logo_data)
        except Exception as e:
            print(f"Error loading logo detection config: {e}")
    
    def _create_palette_from_config(self, config_data: Dict[str, Any]) -> BrandColorPalette:
        """Create BrandColorPalette from configuration data"""
        def parse_colors(color_list: List[Dict[str, Any]]) -> List[BrandColor]:
            colors = []
            for color_data in color_list:
                color = BrandColor(
                    name=color_data['name'],
                    hex_code=color_data['hex'],
                    rgb=color_data.get('rgb'),
                    tolerance=color_data.get('tolerance', 0.1)
                )
                colors.append(color)
            return colors
        
        return BrandColorPalette(
            name=config_data['name'],
            primary_colors=parse_colors(config_data.get('primary_colors', [])),
            secondary_colors=parse_colors(config_data.get('secondary_colors', [])),
            accent_colors=parse_colors(config_data.get('accent_colors', [])),
            forbidden_colors=parse_colors(config_data.get('forbidden_colors', []))
        )
    
    def save_config(self, config_type: str, output_path: str = None):
        """Save configuration to YAML file"""
        if output_path is None:
            output_path = f"{self.config_dir}/{config_type}.yaml"
        
        config_data = {}
        
        if config_type == "color_palette":
            config_data = {
                'brand_palette': {
                    'name': self.color_palette.name,
                    'primary_colors': [
                        {
                            'name': c.name,
                            'hex': c.hex_code,
                            'rgb': c.rgb,
                            'tolerance': c.tolerance
                        } for c in self.color_palette.primary_colors
                    ],
                    'secondary_colors': [
                        {
                            'name': c.name,
                            'hex': c.hex_code,
                            'rgb': c.rgb,
                            'tolerance': c.tolerance
                        } for c in self.color_palette.secondary_colors
                    ],
                    'accent_colors': [
                        {
                            'name': c.name,
                            'hex': c.hex_code,
                            'rgb': c.rgb,
                            'tolerance': c.tolerance
                        } for c in self.color_palette.accent_colors
                    ],
                    'forbidden_colors': [
                        {
                            'name': c.name,
                            'hex': c.hex_code,
                            'rgb': c.rgb,
                            'tolerance': c.tolerance
                        } for c in self.color_palette.forbidden_colors
                    ]
                }
            }
        elif config_type == "typography_rules":
            config_data = {
                'typography_rules': {
                    'approved_fonts': self.typography_rules.approved_fonts,
                    'max_font_size': self.typography_rules.max_font_size,
                    'min_font_size': self.typography_rules.min_font_size,
                    'preferred_font_families': self.typography_rules.preferred_font_families,
                    'forbidden_fonts': self.typography_rules.forbidden_fonts,
                    'line_height_ratio': self.typography_rules.line_height_ratio,
                    'letter_spacing': self.typography_rules.letter_spacing
                }
            }
        elif config_type == "brand_voice":
            config_data = {
                'brand_voice': {
                    'formality_score': self.brand_voice.formality_score,
                    'confidence_level': self.brand_voice.confidence_level,
                    'warmth_score': self.brand_voice.warmth_score,
                    'energy_score': self.brand_voice.energy_score,
                    'readability_level': self.brand_voice.readability_level,
                    'persona_type': self.brand_voice.persona_type,
                    'allow_emojis': self.brand_voice.allow_emojis,
                    'allow_slang': self.brand_voice.allow_slang,
                    'no_financial_guarantees': self.brand_voice.no_financial_guarantees,
                    'no_medical_claims': self.brand_voice.no_medical_claims,
                    'no_competitor_bashing': self.brand_voice.no_competitor_bashing
                }
            }
        elif config_type == "logo_detection":
            config_data = {
                'logo_detection': {
                    'confidence_threshold': self.logo_detection.confidence_threshold,
                    'iou_threshold': self.logo_detection.iou_threshold,
                    'max_detections': self.logo_detection.max_detections,
                    'min_logo_size': self.logo_detection.min_logo_size,
                    'max_logo_size': self.logo_detection.max_logo_size,
                    'placement_rules': self.logo_detection.placement_rules,
                    'enable_llva_ollama': self.logo_detection.enable_llva_ollama,
                    'llva_analysis_focus': self.logo_detection.llva_analysis_focus,
                    'llva_model': self.logo_detection.llva_model,
                    'llva_confidence_threshold': self.logo_detection.llva_confidence_threshold
                }
            }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to {output_path}")

# Default settings instance
settings = Settings()
