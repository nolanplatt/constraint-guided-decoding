# Constraint-Guided Decoding for Small Language Models

Lightweight inference-time constraint enforcement for small language models via a modified beam search with Z3-based semantic verification.

## Overview

This project implements constraint-guided decoding that will enforce arithmetic and logical consistency during text generation without *any* model retraining. Unlike syntactic constrained decoding approaches that enforce grammar/format constraints, this project focuses more on **semantic constraints** — verifying mathematical consistency (e.g., if x=5, then 2x should always equal 10).

## Key Features

- **Inference-time constraint verification** via Z3 SMT solver integration
- **Modified beam search** with constraint-aware pruning (Algorithm 1)
- **4-bit quantized Phi-2** for resource-constrained deployment (<6GB VRAM)
- **Runs on free Google Colab** (T4 GPU)

## Quick Start

### Option 1: Google Colab (Recommended)
1. Upload `constraint_guided_decoding.ipynb` to Google Colab
2. Runtime; change runtime type; T4 GPU
3. Run all cells in order

### Option 2: Local Installation
```bash
git clone https://github.com/nolanplatt/constraint-guided-decoding.git
cd constraint-guided-decoding
pip install -r requirements.txt
jupyter notebook constraint_guided_decoding.ipynb
```

## Requirements

- Python 3.9+
- CUDA-compatible GPU (6GB+ VRAM recommended) or Colab
- Please see `requirements.txt` for dependencies

## Usage

```python
# Initialize
constraint_checker = ConstraintChecker()
decoder = ConstraintGuidedDecoder(
    model=model,
    tokenizer=tokenizer,
    constraint_checker=constraint_checker,
    beam_width=5,
    lambda_weight=1.0
)

# Generate w/ constraints
result = decoder.generate(
    prompt="x = 5. Calculate 2x.\nAnswer: 2x =",
    max_new_tokens=20,
    constraint_types=['arithmetic']
)

print(result['text'])
print(f"Violations: {result['violations']}")
```



## Algorithm

The core algorithm modifies beam search to incorporate constraint verification at each decoding step:

```
score(y_t | y_{<t}) = log P(y_t | x, y_{<t}) + λ · φ(y_{≤t})
```

where φ(y) ∈ {0, -∞} represents constraint satisfaction.

## Limitations

- ~5x computational overhead compared to unconstrained generation
- Regex-based constraint extraction does have edge cases
- Currently limited to arithmetic consistency constraints

## Citation

```bibtex
@misc{platt2025constraint,
  author = {Platt, Nolan W.},
  title = {Constraint-Guided Decoding for Small Language Models},
  year = {2025},
  institution = {Virginia Tech}
}
```

## License

MIT

## Acknowledgments

ECE 4424/CS 4824 Capstone Project, Virginia Tech, Fall 2025
