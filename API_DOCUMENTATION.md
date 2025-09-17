# BrandGuard Consolidated Pipeline API Documentation

## Overview

The BrandGuard Consolidated Pipeline provides a unified API for comprehensive brand analysis including color validation, typography analysis, copywriting assessment, and logo detection. The system uses advanced AI models including YOLOv8 nano, Qwen2.5-VL-3B-Instruct, and various NLP models for accurate brand compliance checking.

## Key Features

### 🎯 **Advanced Text Extraction**
- **VLLM-Powered Analysis**: Uses Qwen2.5-VL-3B-Instruct for intelligent text extraction from images
- **OCR Fallback**: Automatic fallback to PaddleOCR when VLLM extraction fails
- **Multi-Pattern Recognition**: Handles various text formats including quoted text, descriptive text, and mixed content
- **Real-time Processing**: Extracts text content and provides it in API responses

### 🔍 **Comprehensive Brand Analysis**
- **Color Validation**: CIEDE2000 color matching with brand palette compliance
- **Typography Analysis**: Font identification and brand guideline compliance
- **Copywriting Assessment**: Tone analysis, grammar checking, and brand voice validation
- **Logo Detection**: Hybrid YOLOv8 + Qwen detection with placement validation

### 🚀 **Production-Ready Features**
- **Docker Support**: Full containerization with Docker Compose
- **Health Monitoring**: Built-in health checks and status endpoints
- **Error Handling**: Robust error handling with graceful fallbacks
- **Scalable Architecture**: Designed for high-volume production use

## Base URL

```
http://localhost:5001
```

## Authentication

Currently no authentication is required. All endpoints are publicly accessible.

## Content Types

- **File Uploads**: `multipart/form-data`
- **Text Input**: `application/x-www-form-urlencoded`
- **Responses**: `application/json`

## Endpoints

### 1. Health Check

**GET** `/api/health`

Check system health and status.

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2025-09-06T20:00:00.000Z",
  "pipeline_ready": true,
  "settings_loaded": true,
  "version": "1.0.0"
}
```

---

### 2. Comprehensive Analysis

**POST** `/api/analyze`

Perform comprehensive brand analysis using all available models.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_type` | string | No | Input type: `file`, `text`, or `url` (default: `file`) |
| `file` | file | Yes* | Image file to analyze (*required if input_type=file) |
| `text_content` | string | Yes* | Text content to analyze (*required if input_type=text) |
| `url` | string | Yes* | URL to analyze (*required if input_type=url) |

#### Color Analysis Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_color` | boolean | `true` | Enable color analysis |
| `primary_colors` | string | `""` | Comma-separated primary brand colors (hex format) |
| `secondary_colors` | string | `""` | Comma-separated secondary brand colors (hex format) |
| `accent_colors` | string | `""` | Comma-separated accent brand colors (hex format) |
| `primary_threshold` | integer | `75` | Primary color matching threshold (0-100) |
| `secondary_threshold` | integer | `75` | Secondary color matching threshold (0-100) |
| `accent_threshold` | integer | `75` | Accent color matching threshold (0-100) |
| `color_tolerance` | float | `2.3` | CIEDE2000 color difference tolerance |
| `enable_contrast_check` | boolean | `true` | Enable WCAG 2.1 contrast checking |

#### Typography Analysis Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_typography` | boolean | `true` | Enable typography analysis |
| `merge_regions` | boolean | `true` | Merge nearby text regions |
| `distance_threshold` | integer | `20` | Distance threshold for region merging |
| `confidence_threshold` | float | `0.7` | Font identification confidence threshold |
| `expected_fonts` | string | `""` | Comma-separated expected font names |

#### Copywriting Analysis Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_copywriting` | boolean | `true` | Enable copywriting analysis |
| `formality_score` | integer | `60` | Target formality score (0-100) |
| `confidence_level` | string | `balanced` | Confidence level: `conservative`, `balanced`, `aggressive` |
| `warmth_score` | integer | `50` | Target warmth score (0-100) |
| `energy_score` | integer | `50` | Target energy score (0-100) |

#### Logo Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_logo` | boolean | `true` | Enable logo detection |
| `logo_confidence_threshold` | float | `0.5` | Logo detection confidence threshold |
| `max_logo_detections` | integer | `100` | Maximum number of logo detections |
| `enable_placement_validation` | boolean | `true` | Enable logo placement validation |
| `enable_brand_compliance` | boolean | `true` | Enable brand compliance checking |
| `generate_annotations` | boolean | `true` | Generate annotated images |
| `allowed_zones` | string | `top-left,top-right,bottom-left,bottom-right` | Allowed logo placement zones |
| `min_logo_size` | float | `0.01` | Minimum logo size (relative to image) |
| `max_logo_size` | float | `0.25` | Maximum logo size (relative to image) |
| `min_edge_distance` | float | `0.05` | Minimum distance from image edges |

#### Response

```json
{
  "success": true,
  "results": {
    "analysis_id": "uuid-string",
    "overall_score": 0.85,
    "model_results": {
      "color_analysis": {
        "dominant_colors": [
          {
            "hex": "#FF0000",
            "rgb": [255, 0, 0],
            "percentage": 45.2,
            "category": "primary",
            "brand_match": true,
            "similarity_score": 0.92
          }
        ],
        "brand_compliance": {
          "primary_match": true,
          "secondary_match": false,
          "accent_match": true,
          "overall_score": 0.75
        },
        "contrast_check": {
          "wcag_aa_compliant": true,
          "wcag_aaa_compliant": false,
          "contrast_ratio": 4.5
        }
      },
      "typography_analysis": {
        "detected_fonts": [
          {
            "name": "Arial",
            "confidence": 0.95,
            "regions": 3
          }
        ],
        "compliance_score": 0.88,
        "violations": []
      },
      "copywriting_analysis": {
        "extracted_text": "all in or nothing\nMartinez Nitrocharge\nadidas",
        "text_content": "all in or nothing\nMartinez Nitrocharge\nadidas",
        "tone_analysis": {
          "formality": {
            "formality_level": "formal",
            "formality_score": 0.9
          },
          "sentiment": {
            "overall_sentiment": "motivational",
            "compound": 0.0
          },
          "readability": {
            "score": 0.9,
            "level": "grade 8"
          }
        },
        "grammar_analysis": {
          "grammar_score": 95,
          "grammar_errors": [],
          "grammar_suggestions": [],
          "spelling_errors": [],
          "punctuation_issues": []
        },
        "visual_elements": {
          "has_text": true,
          "text_quality": "good",
          "visual_appeal": "high",
          "colors": ["black", "white"],
          "layout": "centered text",
          "branding": "adidas logo visible",
          "text_placement": "bottom right of image"
        },
        "text_metrics": {
          "word_count": 6,
          "sentence_count": 1,
          "readability_level": "grade 8"
        },
        "brand_voice_compliance": {
          "score": 0.95,
          "failures": [],
          "explanations": [],
          "failure_summary": "Text meets requirements"
        },
        "compliance": {
          "score": 0.95,
          "failures": [],
          "explanations": [],
          "failure_summary": "Text meets requirements"
        },
        "copywriting_score": 0.92,
        "errors": [],
        "recommendations": [
          "Text effectively conveys brand message",
          "Visual elements enhance readability"
        ]
      },
      "logo_analysis": {
        "detections": [
          {
            "bbox": [100, 50, 200, 150],
            "confidence": 0.89,
            "brand": "Company Name",
            "placement_valid": true,
            "size_valid": true
          }
        ],
        "validation": {
          "compliance_score": 0.92,
          "valid_placements": 1,
          "violations": []
        },
        "annotated_image": "base64_encoded_image"
      }
    },
    "recommendations": [
      "Consider adjusting secondary colors for better brand compliance",
      "Logo placement meets brand guidelines"
    ],
    "processing_time": 3.2
  }
}
```

---

### 3. Color Analysis Only

**POST** `/api/analyze/color`

Perform color analysis only.

#### Request Parameters

Same as color analysis parameters from comprehensive analysis endpoint.

#### Response

```json
{
  "success": true,
  "color_analysis": {
    "dominant_colors": [...],
    "brand_compliance": {...},
    "contrast_check": {...}
  }
}
```

---

### 4. Typography Analysis Only

**POST** `/api/analyze/typography`

Perform typography analysis only.

#### Request Parameters

Same as typography analysis parameters from comprehensive analysis endpoint.

#### Response

```json
{
  "success": true,
  "typography_analysis": {
    "detected_fonts": [...],
    "compliance_score": 0.88,
    "violations": []
  }
}
```

---

### 5. Copywriting Analysis Only

**POST** `/api/analyze/copywriting`

Perform copywriting analysis only.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_type` | string | Yes | Input type: `file` or `text` |
| `file` | file | Yes* | File containing text (*required if input_type=file) |
| `text_content` | string | Yes* | Text content (*required if input_type=text) |

Plus copywriting analysis parameters from comprehensive analysis endpoint.

#### Response

```json
{
  "success": true,
  "copywriting_analysis": {
    "extracted_text": "all in or nothing\nMartinez Nitrocharge\nadidas",
    "text_content": "all in or nothing\nMartinez Nitrocharge\nadidas",
    "tone_analysis": {
      "formality": {
        "formality_level": "formal",
        "formality_score": 0.9
      },
      "sentiment": {
        "overall_sentiment": "motivational",
        "compound": 0.0
      },
      "readability": {
        "score": 0.9,
        "level": "grade 8"
      }
    },
    "grammar_analysis": {
      "grammar_score": 95,
      "grammar_errors": [],
      "grammar_suggestions": [],
      "spelling_errors": [],
      "punctuation_issues": []
    },
    "visual_elements": {
      "has_text": true,
      "text_quality": "good",
      "visual_appeal": "high",
      "colors": ["black", "white"],
      "layout": "centered text",
      "branding": "adidas logo visible",
      "text_placement": "bottom right of image"
    },
    "text_metrics": {
      "word_count": 6,
      "sentence_count": 1,
      "readability_level": "grade 8"
    },
    "brand_voice_compliance": {
      "score": 0.95,
      "failures": [],
      "explanations": [],
      "failure_summary": "Text meets requirements"
    },
    "compliance": {
      "score": 0.95,
      "failures": [],
      "explanations": [],
      "failure_summary": "Text meets requirements"
    },
    "copywriting_score": 0.92,
    "errors": [],
    "recommendations": [
      "Text effectively conveys brand message",
      "Visual elements enhance readability"
    ]
  }
}
```

---

### 6. Logo Detection Only

**POST** `/api/analyze/logo`

Perform logo detection analysis only.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Image file to analyze |

Plus logo detection parameters from comprehensive analysis endpoint.

#### Response

```json
{
  "success": true,
  "logo_analysis": {
    "detections": [...],
    "validation": {...},
    "annotated_image": "base64_encoded_image"
  }
}
```

---

### 7. Configuration Management

**GET** `/api/config`

Get current system configuration.

#### Response

```json
{
  "success": true,
  "config": {
    "color_palette": {
      "name": "Default Palette",
      "primary_colors_count": 2,
      "secondary_colors_count": 3
    },
    "typography_rules": {
      "approved_fonts_count": 5,
      "max_font_size": 72,
      "min_font_size": 8
    },
    "brand_voice": {
      "formality_score": 60,
      "confidence_level": "balanced",
      "warmth_score": 50,
      "energy_score": 50
    },
    "logo_detection": {
      "confidence_threshold": 0.5,
      "max_detections": 100
    }
  }
}
```

**POST** `/api/config`

Update system configuration.

#### Request Body

```json
{
  "color_palette": {
    "primary_colors": ["#FF0000", "#00FF00"],
    "secondary_colors": ["#0000FF"],
    "accent_colors": ["#FFFF00"]
  },
  "typography_rules": {
    "approved_fonts": ["Arial", "Helvetica", "Times New Roman"]
  },
  "brand_voice": {
    "formality_score": 70,
    "confidence_level": "conservative"
  }
}
```

#### Response

```json
{
  "success": true,
  "message": "Configuration updated successfully"
}
```

---

### 8. Analysis Status

**GET** `/api/status/<analysis_id>`

Get the status of a specific analysis.

#### Response

```json
{
  "success": true,
  "status": {
    "status": "completed",
    "results": {...}
  }
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional error details"
  }
}
```

### Common Error Codes

- `400` - Bad Request (invalid parameters)
- `404` - Not Found (endpoint or resource not found)
- `413` - Payload Too Large (file too big)
- `500` - Internal Server Error
- `503` - Service Unavailable (pipeline not ready)

## Rate Limiting

Currently no rate limiting is implemented. Consider implementing rate limiting for production deployments.

## File Upload Limits

- **Maximum file size**: 50MB
- **Supported image formats**: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- **Supported document formats**: PDF, TXT, DOC, DOCX

## Examples

### Python Example

```python
import requests

# Comprehensive analysis
with open('brand_material.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/api/analyze',
        files={'file': f},
        data={
            'enable_color': 'true',
            'primary_colors': '#FF0000,#00FF00',
            'secondary_colors': '#0000FF',
            'accent_colors': '#FFFF00',
            'primary_threshold': '75',
            'enable_logo': 'true',
            'logo_confidence_threshold': '0.5'
        }
    )

result = response.json()
print(f"Overall Score: {result['results']['overall_score']}")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('enable_color', 'true');
formData.append('primary_colors', '#FF0000,#00FF00');
formData.append('secondary_colors', '#0000FF');
formData.append('accent_colors', '#FFFF00');

fetch('http://localhost:5001/api/analyze', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Analysis Results:', data.results);
    console.log('Overall Score:', data.results.overall_score);
});
```

### cURL Example

```bash
curl -X POST \
  -F "file=@brand_material.jpg" \
  -F "enable_color=true" \
  -F "primary_colors=#FF0000,#00FF00" \
  -F "secondary_colors=#0000FF" \
  -F "accent_colors=#FFFF00" \
  -F "primary_threshold=75" \
  -F "enable_logo=true" \
  -F "logo_confidence_threshold=0.5" \
  http://localhost:5001/api/analyze
```

### Text Extraction Example

```python
import requests

# Analyze image with text extraction
with open('adidas_campaign.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/api/analyze',
        files={'file': f},
        data={
            'enable_copywriting': 'true',
            'enable_logo': 'true'
        }
    )

result = response.json()
copywriting = result['results']['model_results']['copywriting_analysis']

# Access extracted text
print(f"Extracted Text: {copywriting['extracted_text']}")
print(f"Word Count: {copywriting['text_metrics']['word_count']}")
print(f"Formality Level: {copywriting['tone_analysis']['formality']['formality_level']}")
print(f"Text Quality: {copywriting['visual_elements']['text_quality']}")
print(f"Brand Compliance: {copywriting['brand_voice_compliance']['score']}")
```

**Expected Output:**
```
Extracted Text: all in or nothing
Martinez Nitrocharge
adidas
Word Count: 6
Formality Level: formal
Text Quality: good
Brand Compliance: 0.95
```

## Text Extraction Response Fields

### Copywriting Analysis Response Structure

The copywriting analysis now includes comprehensive text extraction and analysis:

| Field | Type | Description |
|-------|------|-------------|
| `extracted_text` | string | Raw text extracted from the image |
| `text_content` | string | Alias for extracted_text (compatibility) |
| `tone_analysis` | object | Detailed tone and sentiment analysis |
| `grammar_analysis` | object | Grammar and spelling analysis |
| `visual_elements` | object | Visual analysis of text presentation |
| `text_metrics` | object | Quantitative text metrics |
| `brand_voice_compliance` | object | Brand voice compliance scoring |
| `compliance` | object | Overall compliance assessment |
| `copywriting_score` | float | Overall copywriting score (0-1) |
| `errors` | array | Any errors encountered during analysis |
| `recommendations` | array | Improvement recommendations |

### Tone Analysis Structure

```json
{
  "tone_analysis": {
    "formality": {
      "formality_level": "formal|professional|casual|very_casual",
      "formality_score": 0.9
    },
    "sentiment": {
      "overall_sentiment": "positive|negative|neutral|motivational",
      "compound": 0.0
    },
    "readability": {
      "score": 0.9,
      "level": "grade 8"
    }
  }
}
```

### Visual Elements Structure

```json
{
  "visual_elements": {
    "has_text": true,
    "text_quality": "good|fair|poor|none",
    "visual_appeal": "high|medium|low",
    "colors": ["black", "white"],
    "layout": "centered text|left-aligned|right-aligned|unknown",
    "branding": "adidas logo visible|none|detected",
    "text_placement": "center|top|bottom|left|right|unknown"
  }
}
```

### Text Metrics Structure

```json
{
  "text_metrics": {
    "word_count": 6,
    "sentence_count": 1,
    "readability_level": "grade 8"
  }
}
```

## Changelog

### Version 1.1.0
- **Enhanced Text Extraction**: Improved VLLM text extraction with multiple pattern recognition
- **OCR Fallback**: Automatic fallback to PaddleOCR when VLLM extraction fails
- **Rich API Response**: Added extracted_text and text_content fields to copywriting analysis
- **Improved Visual Analysis**: Enhanced visual elements detection and text quality assessment
- **Better Error Handling**: Robust error handling with graceful fallbacks for text extraction

### Version 1.0.0
- Initial release with comprehensive brand analysis
- Hybrid logo detection (YOLOv8 nano + Qwen2.5-VL-3B-Instruct)
- Multi-category color validation with CIEDE2000
- Advanced typography analysis
- Copywriting tone analysis with Qwen integration
- RESTful API with comprehensive documentation