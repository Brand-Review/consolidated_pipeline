# Consolidated BrandGuard Pipeline

A unified backend serving all four models: Color, Typography, Copywriting, and Logo Detection.

## Features

### 🎨 Color Analysis
- Extract dominant colors from images
- Validate against brand color palettes
- WCAG contrast compliance checking

### 🔤 Typography Analysis
- Font identification and validation
- Text region detection
- Brand font compliance checking

### ✍️ Copywriting Analysis
- Text extraction and tone analysis
- Brand voice validation
- Compliance scoring

### 🏢 Logo Detection & Analysis
- **YOLOv8-based detection**: Traditional computer vision approach
- **LLVa with Ollama integration**: AI-powered analysis for enhanced accuracy
- **Combined results**: Merges both approaches for comprehensive analysis
- Placement validation and compliance checking
- Bounding box visualization

## LLVa with Ollama Integration

The logo detection now supports a dual approach:

1. **YOLOv8 Detection**: Fast, traditional object detection
2. **LLVa Analysis**: AI-powered understanding of logo context and placement
3. **Combined Results**: Merged analysis for enhanced accuracy

### Configuration

Set environment variables for LLVa integration:
```bash
# Enable LLVa with Ollama
USE_OLLAMA=true
OLLAMA_MODEL=llava-1.5

# Or use HuggingFace MLLM
USE_MLLM=true
MLLM_MODEL=llava-hf/llava-1.5-7b-hf
```

### Analysis Focus Options

- **Comprehensive**: Logo detection + context analysis
- **Logo Only**: Focus on logo detection accuracy
- **Context Only**: Analyze placement and context

## API Endpoints

- `POST /api/analyze` - Main analysis endpoint
- `GET /api/health` - Health check

## Model Settings

Each analysis type has configurable parameters accessible through the web interface:

- Confidence thresholds
- Validation rules
- Analysis focus areas
- Performance settings

## Results Display

The interface shows:
- **Summary**: Overall compliance scores
- **Detailed Results**: Model-specific analysis
- **Model Information**: Which models were used and their performance
- **Visual Analysis**: Bounding boxes and annotations
- **Recommendations**: Improvement suggestions

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Usage

1. Upload content (image, text, or URL)
2. Configure analysis settings
3. Enable desired models
4. Run analysis
5. Review results and recommendations
