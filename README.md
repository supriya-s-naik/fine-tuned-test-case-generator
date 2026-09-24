# Fine-Tuning Qwen3 to Generate Test Cases from User Stories

Week 5 custom project for The Gen Academy.

This project fine-tunes `Qwen/Qwen3-1.7B` with LoRA to convert user stories and numbered acceptance criteria into structured, traceable software test cases. The model returns JSON suitable for a test-management system or automation pipeline.

## Use case

QA engineers repeatedly translate requirements into positive, negative, and boundary tests. The desired output contains exactly these fields:

```json
[
  {
    "id": "TC-001",
    "title": "Verify an available appointment can be booked",
    "type": "positive",
    "preconditions": ["The appointment slot is available"],
    "steps": ["Select the slot", "Confirm the booking"],
    "expected_result": "The appointment is created and confirmation is sent.",
    "covers": ["AC1"]
  }
]
```

Fine-tuning is useful when an organization needs consistent schema, terminology, traceability, and semantic classification across many stories.

## Experiment progression

### Version 1

The first experiment used 60 training and 20 held-out examples. The base model scored 76.2% and the adapter scored 75.7%. Five tuned outputs violated the strict schema.

### Version 2

Version 2 made the JSON contract explicit and raised schema validity to 100%. A canonical prompt reached 96.1%, but the base model reached 96.0% with the same prompt. Investigation revealed a synthetic-data shortcut: AC1 was always positive, AC2 negative, and AC3 boundary. The adapter contribution under that prompt was only +0.1 percentage points.

### Version 3

Version 3 removes the positional shortcut:

- 440 unique examples
- 320 training examples across 16 families
- 60 development examples across three new families
- 60 final-test examples across three different new families
- two to five acceptance criteria per story
- shuffled criterion order
- semantic types independent of AC number
- no family overlap across splits

The notebook evaluates fixed 30-example samples from development and final test. Both models receive the same concise prompt and deterministic decoding settings.

## Version 3 final-test results

| Metric | Base model | Fine-tuned V3 | Change |
|---|---:|---:|---:|
| JSON validity | 100.0% | 100.0% | 0.0 pp |
| Schema validity | 100.0% | 100.0% | 0.0 pp |
| AC coverage | 100.0% | 100.0% | 0.0 pp |
| Type accuracy | 62.8% | 90.8% | **+28.0 pp** |
| ROUGE-L | 54.3% | 65.6% | **+11.3 pp** |
| Overall | 83.4% | 91.3% | **+7.9 pp** |

The Version 3 adapter made a meaningful contribution on three untouched scenario families. It primarily improved semantic test-type classification and alignment with the labelled QA style while preserving perfect structural validity and AC traceability.

## Version 3 training configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` |
| Method | LoRA |
| Dataset | `user_story_test_cases_v3` |
| Chat template | `qwen3_nothink` |
| Learning rate | `1e-5` |
| Epochs | `1.0` |
| Cutoff length | `1024` |
| Batch size | `1` |
| Gradient accumulation | `4` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| Compute type | `fp16` on a T4 GPU |

Training ran for 80 optimization steps. Cross-entropy loss fell from 0.9867 to 0.3287.

## Run Version 3

1. Upload `FineTune_Test_Case_Generator_Qwen3_v3.ipynb` to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the installation cells.
4. Upload `data/user_story_test_cases_v3.csv` when prompted.
5. Train the LoRA adapter in LLaMA Board using the table above.
6. Back up the adapter before deleting or restarting the Colab runtime.
7. Run development evaluation before opening the final-test split.
8. Run the final test once after fixing the configuration.

The notebook produces development and final-test prediction CSVs, summary CSVs, a training curve, and base-versus-tuned charts. Adapter archives and merged model weights are intentionally excluded from Git because they exceed ordinary repository file-size limits.

## Repository contents

| Path | Purpose |
|---|---|
| `FineTune_Test_Case_Generator_Qwen3.ipynb` | Original Version 1 experiment |
| `FineTune_Test_Case_Generator_Qwen3_v2.ipynb` | Schema-focused Version 2 experiment |
| `FineTune_Test_Case_Generator_Qwen3_v3.ipynb` | Recommended semantic Version 3 experiment |
| `data/user_story_test_cases.csv` | Original 80-example dataset |
| `data/user_story_test_cases_v3.csv` | Version 3 dataset with family-level splits |
| `tools/generate_dataset.py` | Original deterministic dataset generator |
| `tools/generate_dataset_v3.py` | Version 3 deterministic dataset generator |
| `tools/build_notebook.py` | Original reproducible notebook builder |
| `tools/build_notebook_v2.py` | Version 2 notebook builder |
| `tools/build_notebook_v3.py` | Version 3 notebook builder |
| `docs/submission_report.md` | Original submission report source |
| `docs/loom_script.md` | Short demonstration script |

## Reproduce generated artifacts

```bash
python tools/generate_dataset.py
python tools/build_notebook.py
python tools/generate_dataset_v3.py
python tools/build_notebook_v3.py
```

The generators are deterministic and overwrite only their corresponding generated artifacts.

## Limitations

The data is synthetic and the final test contains 30 sampled examples. The results demonstrate controlled task adaptation, not production readiness. A production evaluation should use approved real stories, more domains, repeated training seeds, semantic quality review by QA engineers, and checks for duplication, unsupported assumptions, and risk coverage.
