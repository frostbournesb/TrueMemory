# Evaluation Configuration

Configuration used for all LoCoMo benchmark runs.

<p align="center">
  <img src="../../assets/charts/eval-pipeline.png" alt="Evaluation Pipeline" />
</p>

## Answer Generation

| Parameter | Value |
|-----------|-------|
| Model | `openai/gpt-4.1-mini` |
| Temperature | 0 |
| Max Tokens | 200 |

## Judging

| Parameter | Value |
|-----------|-------|
| Model | `openai/gpt-4o-mini` |
| Temperature | 0 |
| Max Tokens | 10 |
| Runs per question | 3 (majority vote) |

## Dataset

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo v1 -- 10 conversations, 1,540 questions |
| Category 1 (single-hop) | 282 questions |
| Category 2 (multi-hop) | 321 questions |
| Category 3 (temporal) | 96 questions |
| Category 4 (open-domain) | 841 questions |
| Category 5 (adversarial) | Excluded per standard practice |

## Preprocessing

| Parameter | Value |
|-----------|-------|
| Temporal resolution | Relative dates resolved to absolutes during ingestion |

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Compute platform | Modal serverless (Python 3.11) |
| API routing | OpenRouter |
| Recommended reproduction platform | Modal (free credits available at [modal.com](https://modal.com)) |
