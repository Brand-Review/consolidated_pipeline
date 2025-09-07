# BrandReviewAI API Documentation

## Overview
BrandReviewAI provides a comprehensive brand compliance analysis API that consolidates four specialized models:
- 🎨 **Color Analysis** - Extract and validate color palettes
- 🔤 **Typography Analysis** - Font identification and validation
- ✍️ **Copywriting Analysis** - Tone and brand voice analysis
- 🏢 **Logo Detection** - Logo detection with placement validation

**Base URL**: `http://localhost:5001` (or your deployed server URL)

---

## 🔍 Health Check

### GET `/api/health`
Check API health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "pipeline_ready": true,
  "settings_loaded": true,
  "version": "1.0.0"
}
```

---

## 🚀 Main Analysis Endpoint

### POST `/api/analyze`
Comprehensive analysis using all enabled models.

**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_type` | string | Yes | `file` | Input type: `file`, `text`, or `url` |
| `file` | file | Conditional* | - | Image/document file (max 50MB) |
| `text_content` | string | Conditional* | - | Direct text input |
| `url` | string | Conditional* | - | URL to analyze |

*Required based on `input_type`

#### Analysis Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analysis_priority` | string | `balanced` | Priority: `balanced`, `color_focused`, `typography_focused`, `copywriting_focused`, `logo_focused` |
| `report_detail` | string | `detailed` | Detail level: `summary`, `detailed`, `comprehensive` |
| `include_recommendations` | boolean | `true` | Include improvement recommendations |

#### Color Analysis Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_color` | boolean | `true` | Enable color analysis |
| `color_n_colors` | integer | `8` | Number of colors to extract |
| `color_tolerance` | float | `0.2` | Color matching tolerance (0.1-0.5) |
| `enable_contrast_check` | boolean | `true` | Enable WCAG contrast analysis |
| `brand_palette` | string | `""` | Comma-separated hex colors (e.g., "#FF0000,#00FF00") |

#### Typography Analysis Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_typography` | boolean | `true` | Enable typography analysis |
| `typography_confidence_threshold` | float | `0.7` | Font detection confidence (0.5-0.9) |
| `merge_regions` | boolean | `true` | Merge nearby text regions |
| `distance_threshold` | integer | `20` | Pixel distance for merging |
| `expected_fonts` | string | `""` | Comma-separated expected fonts |

#### Copywriting Analysis Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_copywriting` | boolean | `true` | Enable copywriting analysis |
| `formality_score` | integer | `60` | Target formality (0-100) |
| `confidence_level` | string | `balanced` | Confidence: `conservative`, `balanced`, `aggressive` |
| `warmth_score` | integer | `50` | Target warmth (0-100) |
| `energy_score` | integer | `50` | Target energy (0-100) |

#### Logo Detection Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_logo` | boolean | `true` | Enable logo detection |
| `logo_confidence_threshold` | float | `0.5` | Detection confidence (0.3-0.9) |
| `enable_placement_validation` | boolean | `true` | Validate logo placement |
| `generate_annotations` | boolean | `true` | Generate bounding boxes |
| `max_logo_detections` | integer | `100` | Maximum logos to detect |
| `allowed_zones` | string | `"top-left,top-right,bottom-left,bottom-right"` | Allowed placement zones |
| `min_logo_size` | float | `0.01` | Minimum logo size (0.001-1.0) |
| `max_logo_size` | float | `0.25` | Maximum logo size (0.001-1.0) |
| `min_edge_distance` | float | `0.05` | Minimum distance from edges |
| `show_original_image` | boolean | `true` | Show original image in results |
| `show_analysis_overlay` | boolean | `true` | Show detection overlay |
| `annotation_style` | string | `bounding_box` | Style: `bounding_box`, `polygon`, `points` |
| `annotation_color` | string | `gray` | Color: `gray`, `red`, `blue`, `green` |
| `pass_threshold` | float | `0.7` | Pass threshold (0.0-1.0) |
| `warning_threshold` | float | `0.5` | Warning threshold (0.0-1.0) |
| `critical_threshold` | float | `0.3` | Critical threshold (0.0-1.0) |
| `enable_llva_ollama` | boolean | `false` | Enable LLVa with Ollama |
| `llva_analysis_focus` | string | `comprehensive` | Focus: `comprehensive`, `logo_only`, `context_only` |

#### Example Request

```javascript
const formData = new FormData();
formData.append('input_type', 'file');
formData.append('file', imageFile);
formData.append('enable_color', 'true');
formData.append('enable_typography', 'true');
formData.append('enable_copywriting', 'true');
formData.append('enable_logo', 'true');
formData.append('color_n_colors', '8');
formData.append('color_tolerance', '0.2');
formData.append('logo_confidence_threshold', '0.5');
formData.append('pass_threshold', '0.7');

fetch('/api/analyze', {
  method: 'POST',
  body: formData
});
```

#### Response Format

```json
{
  "success": true,
  "results": {
    "overall_compliance_score": 0.85,
    "model_results": {
      "color_analysis": {
        "compliance_score": 0.9,
        "dominant_colors": [
          {
            "hex": "#FF0000",
            "rgb": [255, 0, 0],
            "percentage": 25.5
          }
        ],
        "palette_validation": {
          "compliance_score": 0.9,
          "compliant_colors": 7,
          "non_compliant_colors": 1,
          "total_colors": 8
        },
        "contrast_analysis": {
          "wcag_compliance": "AA",
          "overall_score": 0.85
        }
      },
      "typography_analysis": {
        "compliance_score": 0.8,
        "text_regions": 5,
        "font_analysis": [
          {
            "font_family": "Arial",
            "font_size": 16,
            "confidence": 0.95,
            "approved": true,
            "text": "Sample text"
          }
        ],
        "typography_validation": {
          "compliance_score": 0.8,
          "approved_fonts": 4,
          "non_approved_fonts": 1,
          "total_fonts": 5
        }
      },
      "copywriting_analysis": {
        "compliance_score": 0.75,
        "extracted_text": "Sample text content...",
        "tone_analysis": {
          "tone_category": "Professional",
          "formality_score": 75,
          "confidence": 0.88,
          "sentiment_score": 0.6,
          "warmth_score": 45,
          "energy_score": 55
        },
        "brand_voice_validation": {
          "compliance_score": 0.75,
          "formality_match": "Good",
          "confidence_level": "balanced",
          "warmth_match": "Fair"
        }
      },
      "logo_analysis": {
        "compliance_score": 0.9,
        "logo_detections": [
          {
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "confidence": 0.95,
            "class": "logo"
          }
        ],
        "placement_validation": {
          "compliance_score": 0.9,
          "valid_placements": 1,
          "invalid_placements": 0,
          "violations": []
        },
        "brand_compliance": {
          "compliance_score": 0.9,
          "brand_compliant": true
        },
        "analysis_settings": {
          "confidence_threshold": 0.5,
          "model_used": "real_LogoDetector"
        }
      }
    },
    "recommendations": [
      "Consider using more brand-approved fonts",
      "Logo placement follows brand guidelines well"
    ]
  }
}
```

---

## 🎨 Individual Model Endpoints

### POST `/api/analyze/color`
Color analysis only.

**Parameters:**
- `file` (required): Image file
- `n_colors`: Number of colors to extract
- `n_clusters`: Number of clusters for analysis
- `color_tolerance`: Color matching tolerance
- `enable_contrast_check`: Enable contrast analysis

### POST `/api/analyze/typography`
Typography analysis only.

**Parameters:**
- `file` (required): Image file
- `merge_regions`: Merge nearby text regions
- `distance_threshold`: Pixel distance for merging
- `confidence_threshold`: Font detection confidence
- `enable_font_validation`: Enable font validation

### POST `/api/analyze/copywriting`
Copywriting analysis only.

**Parameters:**
- `input_type`: `file` or `text`
- `file`: Image/document file (if input_type is file)
- `text_content`: Direct text (if input_type is text)
- `include_suggestions`: Include improvement suggestions
- `include_industry_benchmarks`: Include industry comparisons
- `enable_brand_profile_matching`: Enable brand voice matching

### POST `/api/analyze/logo`
Logo detection only.

**Parameters:**
- `file` (required): Image file
- `enable_placement_validation`: Validate logo placement
- `enable_brand_compliance`: Check brand compliance
- `generate_annotations`: Generate bounding boxes

---

## ⚙️ Configuration Endpoints

### GET `/api/config`
Get current configuration.

**Response:**
```json
{
  "success": true,
  "config": {
    "color_palette": {
      "name": "Default Brand Palette",
      "primary_colors_count": 5,
      "secondary_colors_count": 3
    },
    "typography_rules": {
      "approved_fonts_count": 8,
      "max_font_size": 72,
      "min_font_size": 12
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

### POST `/api/config`
Update configuration (JSON payload).

---

## 📊 Status Endpoint

### GET `/api/status/<analysis_id>`
Get analysis status by ID.

**Response:**
```json
{
  "success": true,
  "status": {
    "analysis_id": "abc123",
    "status": "completed",
    "progress": 100,
    "estimated_completion": "2024-01-15T10:35:00.000Z"
  }
}
```

---

## 🔧 Error Handling

### Error Response Format
```json
{
  "error": "Error description",
  "details": "Additional error details (optional)"
}
```

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `413`: Payload Too Large (file too big)
- `500`: Internal Server Error
- `503`: Service Unavailable (pipeline not ready)

---

## 📁 File Upload Requirements

### Supported File Types
- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
- **Documents**: PDF, TXT, DOC, DOCX

### File Size Limits
- **Maximum**: 50MB
- **Recommended**: Under 10MB for faster processing

### File Naming
- Files are automatically renamed with timestamps
- Special characters are sanitized
- Original filenames are preserved in metadata

---

## 🚀 Frontend Integration Examples

### Basic File Upload
```javascript
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('input_type', 'file');
  formData.append('file', file);
  
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    if (data.success) {
      console.log('Analysis results:', data.results);
    } else {
      console.error('Analysis failed:', data.error);
    }
  } catch (error) {
    console.error('Network error:', error);
  }
};
```

### Text Analysis
```javascript
const analyzeText = async (text) => {
  const formData = new FormData();
  formData.append('input_type', 'text');
  formData.append('text_content', text);
  formData.append('enable_copywriting', 'true');
  
  const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};
```

### Progress Tracking
```javascript
const trackProgress = async (analysisId) => {
  const response = await fetch(`/api/status/${analysisId}`);
  const data = await response.json();
  
  if (data.success) {
    const { status, progress } = data.status;
    updateProgressBar(progress);
    
    if (status === 'completed') {
      showResults();
    } else if (status === 'processing') {
      setTimeout(() => trackProgress(analysisId), 1000);
    }
  }
};
```

---

## 🔒 Security Considerations

### File Upload Security
- Files are automatically sanitized
- Only allowed extensions are processed
- Files are deleted after analysis
- Maximum file size is enforced

### CORS Support
- Cross-origin requests are supported
- Configure allowed origins in production

### Rate Limiting
- Consider implementing rate limiting for production use
- Monitor API usage and performance

---

## 📱 Mobile Considerations

### File Upload
- Support for mobile camera uploads
- Responsive design for mobile interfaces
- Touch-friendly controls

### Performance
- Optimize image sizes for mobile
- Consider progressive loading for large files
- Implement offline capabilities where possible

---

## 🧪 Testing

### Test Endpoints
- Use `/api/health` to verify API status
- Test with small image files first
- Verify all model combinations work

### Sample Files
- Include test images in your development setup
- Test with various file formats and sizes
- Verify error handling with invalid inputs

---

## 📚 Additional Resources

### Response Schema
- All responses follow consistent JSON structure
- Error responses include descriptive messages
- Success responses include comprehensive data

### Model-Specific Data
- Each model returns specialized analysis results
- Results include confidence scores and validation data
- Recommendations are provided for improvement

### Real-time Updates
- Consider implementing WebSocket connections for live updates
- Use status endpoint for progress tracking
- Implement retry logic for failed requests

---

## 🚀 Getting Started

1. **Start the API server** on port 5001
2. **Test health endpoint** to verify readiness
3. **Upload a test image** using the main analysis endpoint
4. **Review response format** and integrate with your frontend
5. **Implement error handling** for production use
6. **Add progress tracking** for better user experience

For questions or support, refer to the backend logs and error messages for detailed debugging information.
