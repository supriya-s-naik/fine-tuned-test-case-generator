# Fine-Tuning Qwen3 to Generate Test Cases from User Stories

Week 5 custom project for The Gen Academy.

This project fine-tunes `Qwen/Qwen3-1.7B` to turn a user story and numbered acceptance criteria into structured, traceable software test cases. The model returns JSON that can be consumed by a test-management system or automation pipeline.

## The use case

QA engineers repeatedly convert requirements into positive, negative, and boundary test cases. A general model can do this with prompting, but its output structure and requirements coverage can vary. Fine-tuning is useful when an organization needs the same schema, terminology, and traceability rules on every story.

Input:

```text
User Story:
As a shopper, I want to manage products in my cart, so that I can review an accurate order before checkout.

Acceptance Criteria:
AC1: Given an in-stock product, when it is added to the cart, then the item, quantity, and current price appear in the cart.
AC2: Given a product with zero stock, when it is added, then the cart is unchanged and an out-of-stock message appears.
AC3: Given a product in the cart, when quantity is changed to the maximum of 10, then the cart accepts it; quantity 99 is rejected.
```

Output:

```json
[
  {
    "id": "TC-001",
    "title": "Verify AC1: an in-stock product is added",
    "type": "positive",
    "preconditions": ["The product is in stock"],
    "steps": ["Open the product", "Add it to the cart", "Open the cart"],
    "expected_result": "The item, quantity, and current price appear in the cart.",
    "covers": ["AC1"]
  }
]
```

## What the experiment does

1. Generates a deterministic labelled dataset across authentication, commerce, finance, healthcare, collaboration, analytics, travel, and learning domains.
2. Keeps complete scenario families out of training to reduce train/validation leakage.
3. Converts the training set to ShareGPT format for LLaMA Factory.
4. Fine-tunes the 1.7B Qwen3 base model with LoRA.
5. Merges the adapter into a standalone model.
6. Runs the base and fine-tuned models on the same 20 held-out examples.
7. Compares JSON validity, schema validity, acceptance-criteria coverage, test-type coverage, and ROUGE-L.

## Run it step by step

### 1. Open the notebook in Colab

Upload [FineTune_Test_Case_Generator_Qwen3.ipynb](FineTune_Test_Case_Generator_Qwen3.ipynb) to [Google Colab](https://colab.research.google.com/). Choose **Runtime → Change runtime type → T4 GPU**.

### 2. Install the training stack

Run the installation and GPU-check cells. The notebook clones LLaMA Factory and installs its PyTorch dependencies. The GPU check should print the name of the assigned accelerator.

### 3. Upload the dataset

When prompted, upload [data/user_story_test_cases.csv](data/user_story_test_cases.csv). The preparation cell checks the schema, confirms that scenario families do not overlap, writes the ShareGPT training file, and registers `user_story_test_cases` in LLaMA Factory.

### 4. Train in LLaMA Board

Run the web UI cell and open its public Gradio URL. Use these first-run settings:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` |
| Finetuning method | `LoRA` |
| Dataset | `user_story_test_cases` |
| Template | `qwen3_nothink` |
| Cutoff length | `1024` |
| Learning rate | `1e-5` |
| Epochs | `2.0` |
| Batch size | `1` |
| Gradient accumulation | `4` |
| LoRA rank | `8` |
| Compute type on a T4 | `fp16` |

Start training. Wait for **Training completed**, copy the **Output Dir**, and stop the notebook cell using the square stop button.

Before stopping or restarting the Colab runtime, use the Colab Files sidebar to download the completed adapter directory as a backup. Colab stores `/content` temporarily and clears it when the runtime resets.

### 5. Inspect training and set the adapter path

Paste the copied directory into `ADAPTER_DIR` in Step 5. Run the cell. A healthy run generally shows loss trending down. Loss alone is not proof of generalization, so continue to validation.

### 6. Baseline, merge, and smoke test

Run Step 6. It first evaluates the unchanged base model, then attaches and merges the LoRA adapter. The smoke test prints one held-out input and output so malformed JSON or a wrong adapter path is caught early.

### 7. Evaluate

Run Steps 7 and 8. The notebook evaluates all 20 held-out examples and creates:

- `training_curve.png`
- `baseline_vs_finetuned.png`
- `evaluation_predictions.csv`
- `evaluation_summary.csv`
- the merged model at `/content/qwen3_test_case_generator`

The metrics have different meanings. JSON and schema validity measure whether the output can enter a downstream system. Acceptance-criteria coverage measures traceability. Test-type coverage checks positive, negative, and boundary breadth. ROUGE-L is a reference-similarity signal and should not be treated as the sole measure of correctness.

### 8. Capture submission evidence

Run the final notebook cell and take a screenshot showing **EXPERIMENT COMPLETE**, the base-versus-fine-tuned table, the held-out sample count, and the overall delta. Save the two charts as additional evidence.

## Repository contents

| Path | Purpose |
|---|---|
| `FineTune_Test_Case_Generator_Qwen3.ipynb` | Colab training and evaluation workflow |
| `data/user_story_test_cases.csv` | 80 labelled examples with fixed train/validation splits |
| `tools/generate_dataset.py` | Reproducible dataset generator |
| `tools/build_notebook.py` | Reproducible notebook builder |
| `docs/submission_report.md` | Report text to personalize after the run |
| `docs/loom_script.md` | Short walkthrough script for a custom-project submission |

## Limitations

The dataset is synthetic and intentionally small, so the experiment demonstrates task adaptation rather than production readiness. Structural metrics do not prove that every generated test is logically sufficient. Before deployment, a QA engineer should review outputs, the dataset should be expanded with real approved stories, and evaluation should include human ratings for correctness, duplication, risk coverage, and unsupported assumptions.

## Experiment result

The conversational base model scored 76.2% on the composite evaluation and the fine-tuned model scored 75.7%, a change of -0.5 percentage points. Fine-tuning slightly improved test-type coverage and ROUGE-L, preserved 100% JSON validity and acceptance-criteria coverage, and reduced strict schema validity from 80% to 75%. All five tuned structural failures came from the held-out appointment-booking family, where the model returned `expected_result` as a useful array rather than the required string. The result indicates that this small synthetic dataset does not yet justify fine-tuning over the conversational base model.

## Reproduce the dataset locally

```bash
python tools/generate_dataset.py
python tools/build_notebook.py
```

Both scripts are deterministic and overwrite only their corresponding generated artifact.
