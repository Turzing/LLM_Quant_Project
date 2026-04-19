# LLM Explanation Consistency in Quantitative Trading

This repository contains the code and data for the paper:  
**"Can Large Language Models Explain Their Trading Decisions? A Human Evaluation of LLaMA-2 with Stress Tests and Confidence Calibration"**  

## Overview

We evaluate the **explanation-decision consistency** of LLaMA-2-7B on a quantitative trading task. Key contributions:

- Define a new task: explanation-decision consistency for trading signals.
- Propose **EDCS (Explanation-Decision Consistency Score)** with human evaluation (1–5 scale).
- Test model robustness under **adversarial (contradictory) signals**.
- Compare LLaMA-2 with **human baselines** (3 non-expert traders, 360 samples).
- Analyze self-reported **confidence calibration** (optional).

## Requirements

- python 3.10
- pytorch 2.1.0
- pandas 2.1.4
- numpy 1.24.3
- matplotlib 3.8.2
- seaborn 0.13.1
- scipy 1.11.4

## Training

Training is run using:
```
python download_model.py
python step1_download_data.py
python step2_sample.py
python step3_build_prompts.py
python step4_llm_inference.py
python step5_merge_human.py
python step6_prepare_scoring_full.py
python step8_analysis_full.py
```

## Testing

Testing is run using:
```
python step7_score_cli.py
```

## Key Results (from our experiments)

| Model / Condition | EDCS (mean ± std) | Sample Size | Key Finding |
|------------------|-------------------|-------------|--------------|
| Human baseline | 3.43 ± 0.80 | 360 | Reference |
| LLaMA-2 (original prompts) | 3.07 ± 0.50 | 120 | Significantly lower than human (p<0.001) |
| LLaMA-2 + Chain-of-Thought | 3.43 ± 0.64 | 60 | On par with human |
| LLaMA-2 (adversarial signals) | 2.81 ± 0.26 | 30 | Severe degradation |

**Additional findings**:
- Contradiction detection score: **1.17/5** (near failure)
- Trade-off rationality score: **1.67/5**
- Failure rate (score <3) on adversarial inputs: **96.7%**

**Overall**: LLMs can generate fluent and complete explanations (comparable to humans in clarity and completeness) but lack logical consistency and collapse under contradictory signals.

## Repository Structure
## Contributors

- **wenyu zhang** – project lead, methodology, code, writing
- **Weihong Xu** – human evaluation, data annotation, validation
- **Yu Huang** – human evaluation, data annotation, validation
