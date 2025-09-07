# Complete Brand Compliance Detection Flow

This document provides a comprehensive flow diagram that includes all components of the BrandGuard pipeline, including the missing elements you mentioned: K-means + deltaE for color analysis, tone detection, typography detection, and vLLM for inference.

## Complete Detection Flow Diagram

```mermaid
flowchart TD
    A[Upload Image/Text] --> B[Input Preprocessing]
    B --> C{Input Type?}
    
    C -->|Image| D[Image Preprocessing]
    C -->|Text| E[Text Preprocessing]
    C -->|Document| F[Document Preprocessing]
    
    %% Image Analysis Branch
    D --> G[YOLOv8 Nano Detection]
    G --> H{Objects Found?}
    
    H -->|Yes| I[Convert to Logo Format]
    H -->|No| J[Qwen2.5-VL-3B-Instruct Analysis]
    J --> K[Logo Detection & Analysis]
    
    I --> L[Brand Compliance Validation]
    K --> L
    
    %% Color Analysis Branch
    D --> M[Color Analysis Pipeline]
    M --> N[K-means Clustering]
    N --> O[Extract Dominant Colors]
    O --> P[CIEDE2000 DeltaE Calculation]
    P --> Q[Brand Color Matching]
    Q --> R[WCAG Contrast Check]
    R --> S[Color Compliance Score]
    
    %% Typography Analysis Branch
    D --> T[Typography Analysis Pipeline]
    T --> U[Text Region Detection]
    U --> V[Font Identification]
    V --> W[Font Classification]
    W --> X[Typography Rules Validation]
    X --> Y[Typography Compliance Score]
    
    %% Copywriting Analysis Branch
    E --> Z[Copywriting Analysis Pipeline]
    F --> Z
    Z --> AA[Text Extraction]
    AA --> BB[Qwen2.5-VL-3B-Instruct Tone Analysis]
    BB --> CC[Brand Voice Validation]
    CC --> DD[Formality Score Analysis]
    DD --> EE[Sentiment Analysis]
    EE --> FF[Copywriting Compliance Score]
    
    %% vLLM Integration
    BB --> GG[vLLM Server]
    GG --> HH[Qwen2.5-VL-3B-Instruct Model]
    HH --> II[Vision-Language Processing]
    II --> JJ[Structured JSON Response]
    JJ --> BB
    
    %% Convergence and Final Analysis
    L --> KK[Comprehensive Analysis Engine]
    S --> KK
    Y --> KK
    FF --> KK
    
    KK --> LL[Overall Compliance Score]
    LL --> MM[Generate Recommendations]
    MM --> NN[Create Compliance Report]
    NN --> OO[Return Results]
    
    %% Styling
    classDef preprocessing fill:#e1f5fe
    classDef detection fill:#f3e5f5
    classDef analysis fill:#e8f5e8
    classDef vllm fill:#fff3e0
    classDef validation fill:#fce4ec
    classDef output fill:#f1f8e9
    
    class A,B,C,D,E,F preprocessing
    class G,H,I,J,K detection
    class M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,BB,CC,DD,EE,FF analysis
    class GG,HH,II,JJ vllm
    class L,KK,LL,MM,NN validation
    class OO output
```

## Detailed Component Breakdown

### 1. **Color Analysis Pipeline**
- **K-means Clustering**: Groups similar colors in the image
- **CIEDE2000 DeltaE**: Calculates perceptual color differences
- **Brand Color Matching**: Compares extracted colors with brand palette
- **WCAG Contrast Check**: Validates accessibility compliance

### 2. **Typography Analysis Pipeline**
- **Text Region Detection**: Identifies text areas in the image
- **Font Identification**: Recognizes font families and styles
- **Typography Rules Validation**: Checks against brand guidelines
- **Compliance Scoring**: Generates typography compliance metrics

### 3. **Copywriting Analysis Pipeline**
- **Text Extraction**: Extracts text from images/documents
- **Qwen2.5-VL-3B-Instruct**: Advanced tone and sentiment analysis
- **Brand Voice Validation**: Ensures content matches brand personality
- **Multi-dimensional Analysis**: Formality, sentiment, warmth, energy scores

### 4. **Logo Detection Pipeline**
- **YOLOv8 Nano**: Fast object detection for logos
- **Qwen2.5-VL-3B-Instruct**: Advanced logo analysis and context understanding
- **Hybrid Approach**: Combines speed of YOLOv8 with accuracy of Qwen
- **Placement Validation**: Checks logo positioning and sizing

### 5. **vLLM Integration**
- **vLLM Server**: High-performance inference server
- **Qwen2.5-VL-3B-Instruct Model**: Vision-language model for complex analysis
- **Structured JSON Response**: Consistent output format
- **GPU Acceleration**: Optimized for speed and efficiency

### 6. **Comprehensive Analysis Engine**
- **Multi-model Integration**: Combines all analysis results
- **Weighted Scoring**: Calculates overall compliance score
- **Recommendation Generation**: Provides actionable insights
- **Report Creation**: Comprehensive compliance documentation

## Key Features

✅ **Complete Color Analysis**: K-means + CIEDE2000 deltaE calculation  
✅ **Advanced Typography Detection**: Font identification and validation  
✅ **Sophisticated Tone Analysis**: Qwen2.5-VL-3B-Instruct integration  
✅ **Hybrid Logo Detection**: YOLOv8 nano + Qwen2.5-VL-3B-Instruct  
✅ **vLLM Integration**: High-performance inference server  
✅ **Comprehensive Validation**: Multi-dimensional compliance checking  
✅ **Structured Output**: JSON-based results and recommendations  

## Technical Stack

- **Color Analysis**: scikit-learn K-means, scikit-image CIEDE2000
- **Typography**: OpenCV, Tesseract OCR, Font identification models
- **Tone Analysis**: Qwen2.5-VL-3B-Instruct via vLLM
- **Logo Detection**: YOLOv8 nano + Qwen2.5-VL-3B-Instruct
- **Inference**: vLLM server for optimized model serving
- **Integration**: RESTful API with comprehensive error handling

This complete flow diagram now includes all the components you mentioned and provides a comprehensive view of the entire BrandGuard pipeline architecture.
