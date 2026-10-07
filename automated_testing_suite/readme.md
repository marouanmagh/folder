# Automated Testing & Resource Profiling Suite

This folder contains a set of advanced Python automation tools developed to rigorously test, validate, and monitor the performance of complex software systems (local Large Language Models). 

These scripts demonstrate my proficiency in building modular R&D testing frameworks, automating quality assurance (QA) pipelines, and evaluating computational resource efficiency.

## 📁 Repository Contents

### 1. `automated_performance_evaluator.py`
A comprehensive continuous validation engine designed to automate complex refactoring tasks and assess code generation quality.
* **Automated QA & Auto-Correction:** Implements an active auto-correction loop that automatically parses execution errors and prompts the system for code repairs.
* **Sandboxed Execution:** Dynamically creates, builds, and executes isolated Python test environments using `subprocess` and `tempfile` modules to safely validate output.
* **Native API Integration:** Interfaces seamlessly with local APIs to stream completions and extract native performance metrics (Tokens/s, Time To First Token).

### 2. `resource_efficiency_profiler.py`
An advanced profiling script dedicated to hardware resource monitoring and software stress testing.
* **Hardware Telemetry & Sustainability:** Utilizes background threading alongside `psutil` and `pynvml` to actively monitor and log CPU usage, GPU utilization, power consumption (Watts), and VRAM allocation in real-time.
* **Stress Testing:** Evaluates thread-safe state management, rollback mechanics, and system stability under heavy context loads (up to 12,000 tokens).
* **Automated Reporting:** Automatically aggregates test results and hardware telemetry into structured datasets (`.csv`, `.json`) for further data analysis and strategic decision-making.
