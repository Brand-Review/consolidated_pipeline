"""
Measure wall-clock time for YOLO + k-means, sequential vs. concurrent.
Usage: python scripts/time_step12.py <path/to/test_image.jpg>
"""
import sys
import os
import time
import cv2
import logging

# Ensure repo root is on the path regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)  # suppress model chatter

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample.jpg"

from src.brandguard.core.color_analyzer import ColorAnalyzer
from src.brandguard.core.logo_analyzer import LogoAnalyzer
from src.brandguard.core.model_imports import import_all_models, get_imported_models
from concurrent.futures import ThreadPoolExecutor

import_all_models()
models = get_imported_models()


class _Settings:
    pass


color_analyzer = ColorAnalyzer(_Settings(), models)
logo_analyzer = LogoAnalyzer(_Settings(), models)

image = cv2.imread(IMAGE_PATH)
assert image is not None, f"Could not load {IMAGE_PATH}"

RUNS = 3

# --- Sequential baseline ---
seq_times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    logo_analyzer.analyze_logos(image.copy(), {}, rag_context="", few_shot_examples=[])
    color_analyzer.analyze_colors(image, {})
    seq_times.append(time.perf_counter() - t0)

# --- Concurrent ---
con_times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_logo = ex.submit(logo_analyzer.analyze_logos, image.copy(), {}, rag_context="", few_shot_examples=[])
        f_color = ex.submit(color_analyzer.analyze_colors, image, {})
        f_logo.result()
        f_color.result()
    con_times.append(time.perf_counter() - t0)

seq_avg = sum(seq_times) / RUNS
con_avg = sum(con_times) / RUNS
print(f"Sequential avg : {seq_avg * 1000:.1f} ms")
print(f"Concurrent avg : {con_avg * 1000:.1f} ms")
print(f"Speedup        : {seq_avg / con_avg:.2f}x")
