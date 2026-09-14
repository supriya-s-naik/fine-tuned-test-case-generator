"""Build the Colab notebook from readable source cells."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FineTune_Test_Case_Generator_Qwen3.ipynb"


def lines(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        """
        # Fine-Tune Qwen3 to Generate Test Cases from User Stories

        **Week 5 custom project — The Gen Academy**

        This notebook fine-tunes `Qwen/Qwen3-1.7B` with a LoRA adapter. The model receives a user story plus numbered acceptance criteria and returns a JSON array of traceable software test cases.

        ## What success means

        A useful output must:

        1. parse as JSON;
        2. follow the required test-case schema;
        3. cover every acceptance-criterion ID;
        4. include positive, negative, and boundary testing;
        5. resemble the labelled reference answer.

        We measure the base model and fine-tuned model on the same held-out scenario families. This tells us whether fine-tuning adds value beyond prompting alone.
        """
    ),
    markdown(
        """
        ## Step 1 — Start a GPU runtime

        In Colab, choose **Runtime → Change runtime type → T4 GPU**, then run the cells in order. A GPU is required for practical training time.
        """
    ),
    markdown("## Step 2 — Install LLaMA Factory and dependencies"),
    code(
        """
        %cd /content/
        %rm -rf LLaMA-Factory
        !git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
        %cd /content/LLaMA-Factory
        !pip install -q -e ".[torch,bitsandbytes]"
        !pip install -q pandas matplotlib seaborn scikit-learn
        """
    ),
    code(
        """
        import torch

        assert torch.cuda.is_available(), "Enable a GPU: Runtime > Change runtime type > T4 GPU"
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA ready:", torch.cuda.is_available())
        """
    ),
    markdown(
        """
        ## Step 3 — Upload and prepare the labelled dataset

        Upload `data/user_story_test_cases.csv` from this repository. The CSV already contains a fixed `train`/`validation` split. Complete scenario families are held out, which prevents near-duplicate variants from leaking into validation.

        This cell converts the 60 training rows to ShareGPT format and registers the dataset with LLaMA Factory under the name `user_story_test_cases`. The 20 validation rows remain untouched until evaluation.
        """
    ),
    code(
        r'''
        import json
        import pandas as pd
        from google.colab import files
        from pathlib import Path

        LLAMA_DATA_DIR = Path("/content/LLaMA-Factory/data")
        TRAIN_JSON_PATH = LLAMA_DATA_DIR / "user_story_test_cases_train.json"
        DATASET_INFO = LLAMA_DATA_DIR / "dataset_info.json"
        VAL_CSV = Path("/content/test_case_validation.csv")

        SYSTEM_PROMPT = """You are a senior software QA engineer. Given a user story and numbered acceptance criteria, return only a valid JSON array of test cases. Every object must use exactly these fields: id, title, type, preconditions, steps, expected_result, covers. type must be positive, negative, or boundary. preconditions, steps, and covers must be arrays of strings. Include coverage for every acceptance-criterion ID. Do not use markdown or commentary."""

        def format_user(story, criteria):
            return f"User Story:\n{story}\n\nAcceptance Criteria:\n{criteria}"

        print("Upload data/user_story_test_cases.csv")
        uploaded = files.upload()
        csv_path = next(iter(uploaded))
        df = pd.read_csv(csv_path)

        required_columns = {
            "example_id", "scenario_family", "domain", "split",
            "user_story", "acceptance_criteria", "expected_test_cases",
        }
        assert required_columns.issubset(df.columns), f"Missing: {required_columns - set(df.columns)}"
        assert set(df["split"]) == {"train", "validation"}

        df_train = df[df["split"] == "train"].reset_index(drop=True)
        df_val = df[df["split"] == "validation"].reset_index(drop=True)
        overlap = set(df_train["scenario_family"]) & set(df_val["scenario_family"])
        assert not overlap, f"Scenario leakage detected: {overlap}"

        # Fail early if a labelled answer is not valid JSON.
        for raw in df["expected_test_cases"]:
            assert isinstance(json.loads(raw), list)

        records = [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_user(row.user_story, row.acceptance_criteria)},
                    {"role": "assistant", "content": row.expected_test_cases},
                ]
            }
            for row in df_train.itertuples(index=False)
        ]
        TRAIN_JSON_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

        info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
        info["user_story_test_cases"] = {
            "file_name": TRAIN_JSON_PATH.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        DATASET_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        df_val.to_csv(VAL_CSV, index=False)

        print(f"Dataset: {len(df)} examples")
        print(f"Training: {len(df_train)} | Held-out validation: {len(df_val)}")
        print("Training families:", sorted(df_train.scenario_family.unique()))
        print("Validation families:", sorted(df_val.scenario_family.unique()))
        print("Registered dataset name: user_story_test_cases")
        '''
    ),
    markdown(
        """
        ## Step 4 — Train the LoRA adapter in LLaMA Board

        Run the next cell and open the public Gradio link. In the **Train** tab use:

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
        | Compute type on T4 | `fp16` |

        Start training and wait for **Training completed**. Copy the displayed **Output Dir**. Then stop this notebook cell using the square stop button; the web server otherwise keeps running.

        Why LoRA? The base model remains frozen while small low-rank matrices learn the new output behavior. This makes the experiment much cheaper than updating all 1.7 billion parameters.
        """
    ),
    code(
        """
        %cd /content/LLaMA-Factory/
        !GRADIO_SHARE=1 llamafactory-cli webui
        """
    ),
    markdown(
        """
        ## Step 5 — Inspect the training curve

        Paste the Output Dir from LLaMA Board below. Falling loss indicates the adapter is learning the labelled response pattern. We still need held-out evaluation because low training loss alone can mean memorization.
        """
    ),
    code(
        r"""
        import json
        import matplotlib.pyplot as plt
        from pathlib import Path

        ADAPTER_DIR = "/content/LLaMA-Factory/saves/Qwen3-1.7B/lora/train_XXXX-XX-XX-XX-XX-XX"  # CHANGE ME
        MERGED_DIR = "/content/qwen3_test_case_generator"
        BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"

        adapter_path = Path(ADAPTER_DIR)
        assert adapter_path.exists(), f"Update ADAPTER_DIR; folder not found: {adapter_path}"

        # LLaMA Factory releases use either trainer_log.jsonl or Hugging Face's
        # trainer_state.json. Support both so the notebook remains reproducible.
        legacy_log = adapter_path / "trainer_log.jsonl"
        state_candidates = [adapter_path / "trainer_state.json", *adapter_path.glob("checkpoint-*/trainer_state.json")]
        state_file = next((path for path in state_candidates if path.exists()), None)
        if legacy_log.exists():
            log_rows = [json.loads(line) for line in legacy_log.read_text().splitlines() if line.strip()]
            log_source = legacy_log
        elif state_file is not None:
            log_rows = json.loads(state_file.read_text())["log_history"]
            log_source = state_file
        else:
            available = "\n".join(str(path.relative_to(adapter_path)) for path in adapter_path.rglob("*") if path.is_file())
            raise FileNotFoundError(f"No supported training log in {adapter_path}. Files found:\n{available}")

        loss_rows = [row for row in log_rows if row.get("loss") is not None]
        steps = [row.get("current_steps", row.get("step")) for row in loss_rows]
        losses = [row["loss"] for row in loss_rows]
        assert losses, f"No loss entries found in {log_source}"

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, losses, color="#5B4BCE", linewidth=2)
        ax.set(title="LoRA training loss", xlabel="Step", ylabel="Cross-entropy loss")
        ax.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig("/content/training_curve.png", dpi=150)
        plt.show()
        print("Log source:", log_source)
        print(f"Starting loss: {losses[0]:.4f} | Final loss: {losses[-1]:.4f} | Drop: {losses[0]-losses[-1]:.4f}")
        """
    ),
    markdown(
        """
        ## Step 6 — Run the base-model baseline, merge LoRA, and smoke-test

        The baseline uses the same prompt, decoding settings, and held-out rows as the tuned model. After measuring it, we attach the adapter and merge its learned deltas into the base weights. The merged directory is a standalone model checkpoint.

        Generation is slower than classification. The 20-row baseline may take several minutes on a T4.
        """
    ),
    code(
        r'''
        import gc
        import re
        import torch
        import pandas as pd
        from pathlib import Path
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        df_val = pd.read_csv("/content/test_case_validation.csv")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        saved_chat_template = Path(ADAPTER_DIR) / "chat_template.jinja"
        if saved_chat_template.exists():
            tokenizer.chat_template = saved_chat_template.read_text(encoding="utf-8")
            print("Using training chat template:", saved_chat_template)

        def make_prompt(story, criteria):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user(story, criteria)},
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        def generate_cases(model, story, criteria):
            prompt = make_prompt(story, criteria)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print("Loading base model and running baseline...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        base_model.eval()
        baseline_outputs = [
            generate_cases(base_model, row.user_story, row.acceptance_criteria)
            for row in tqdm(df_val.itertuples(index=False), total=len(df_val), desc="Base model")
        ]

        print("Attaching and merging LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        tuned_model = peft_model.merge_and_unload()
        tuned_model.eval()
        Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
        tuned_model.save_pretrained(MERGED_DIR)
        tokenizer.save_pretrained(MERGED_DIR)
        _model = tuned_model
        del peft_model, base_model
        gc.collect()
        torch.cuda.empty_cache()
        print("Merged model saved to", MERGED_DIR)

        # Fast sanity check before the full evaluation.
        smoke = df_val.iloc[0]
        smoke_output = generate_cases(_model, smoke.user_story, smoke.acceptance_criteria)
        print("\nSMOKE TEST INPUT\n", format_user(smoke.user_story, smoke.acceptance_criteria))
        print("\nSMOKE TEST OUTPUT\n", smoke_output)
        '''
    ),
    markdown(
        """
        ## Step 7 — Evaluate the tuned model

        We deliberately avoid plain exact-match accuracy: several differently worded test cases can all be correct. Instead we use five repeatable signals:

        - **JSON validity** — can downstream tooling parse the answer?
        - **Schema validity** — are all required fields and allowed types present?
        - **AC coverage** — does `covers` include every acceptance-criterion ID?
        - **Test-type coverage** — are positive, negative, and boundary cases present?
        - **ROUGE-L** — how closely does the wording and structure match the labelled reference?
        """
    ),
    code(
        r'''
        tuned_outputs = [
            generate_cases(_model, row.user_story, row.acceptance_criteria)
            for row in tqdm(df_val.itertuples(index=False), total=len(df_val), desc="Fine-tuned model")
        ]
        print(f"Generated {len(tuned_outputs)} held-out predictions.")
        '''
    ),
    code(
        r'''
        from collections import Counter
        import numpy as np

        REQUIRED_FIELDS = {"id", "title", "type", "preconditions", "steps", "expected_result", "covers"}
        ALLOWED_TYPES = {"positive", "negative", "boundary"}

        def parse_json_array(text):
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", text, flags=re.S)
                if not match:
                    return None
                try:
                    value = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            return value if isinstance(value, list) else None

        def schema_is_valid(cases):
            if not cases:
                return False
            for case in cases:
                if not isinstance(case, dict) or set(case) != REQUIRED_FIELDS:
                    return False
                if case["type"] not in ALLOWED_TYPES:
                    return False
                if not all(isinstance(case[k], list) and case[k] for k in ("preconditions", "steps", "covers")):
                    return False
                if not all(isinstance(case[k], str) and case[k].strip() for k in ("id", "title", "expected_result")):
                    return False
            return True

        def rouge_l(reference, prediction):
            ref = re.findall(r"\w+", reference.lower())
            pred = re.findall(r"\w+", prediction.lower())
            if not ref or not pred:
                return 0.0
            previous = [0] * (len(pred) + 1)
            for ref_token in ref:
                current = [0]
                for j, pred_token in enumerate(pred, start=1):
                    current.append(previous[j - 1] + 1 if ref_token == pred_token else max(previous[j], current[-1]))
                previous = current
            lcs = previous[-1]
            precision, recall = lcs / len(pred), lcs / len(ref)
            return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        def score_one(raw_output, reference, criteria):
            cases = parse_json_array(raw_output)
            expected_ids = set(re.findall(r"\bAC\d+\b", criteria))
            if cases is None:
                return {"json_valid": 0, "schema_valid": 0, "ac_coverage": 0, "type_coverage": 0, "rouge_l": rouge_l(reference, raw_output)}
            covered = {
                ac for case in cases if isinstance(case, dict)
                for ac in case.get("covers", []) if isinstance(ac, str)
            }
            seen_types = {
                case.get("type") for case in cases if isinstance(case, dict)
            }
            return {
                "json_valid": 1,
                "schema_valid": int(schema_is_valid(cases)),
                "ac_coverage": len(covered & expected_ids) / len(expected_ids) if expected_ids else 1,
                "type_coverage": len(seen_types & ALLOWED_TYPES) / len(ALLOWED_TYPES),
                "rouge_l": rouge_l(reference, raw_output),
            }

        def evaluate(outputs, model_name):
            rows = []
            for source, output in zip(df_val.itertuples(index=False), outputs):
                metrics = score_one(output, source.expected_test_cases, source.acceptance_criteria)
                rows.append({"example_id": source.example_id, "model": model_name, **metrics, "output": output})
            return pd.DataFrame(rows)

        base_results = evaluate(baseline_outputs, "Base model")
        tuned_results = evaluate(tuned_outputs, "Fine-tuned")
        all_results = pd.concat([base_results, tuned_results], ignore_index=True)
        metric_columns = ["json_valid", "schema_valid", "ac_coverage", "type_coverage", "rouge_l"]
        summary = all_results.groupby("model")[metric_columns].mean().loc[["Base model", "Fine-tuned"]]
        summary["overall"] = summary.mean(axis=1)
        display(summary.style.format("{:.1%}").background_gradient(cmap="Purples", axis=0))
        '''
    ),
    code(
        r'''
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["JSON valid", "Schema valid", "AC coverage", "Type coverage", "ROUGE-L", "Overall"]
        columns = ["json_valid", "schema_valid", "ac_coverage", "type_coverage", "rouge_l", "overall"]
        x = np.arange(len(labels))
        width = .36

        fig, ax = plt.subplots(figsize=(11, 5))
        base_bars = ax.bar(x - width/2, summary.loc["Base model", columns], width, label="Base model", color="#B8B3E9")
        tuned_bars = ax.bar(x + width/2, summary.loc["Fine-tuned", columns], width, label="Fine-tuned", color="#5B4BCE")
        ax.bar_label(base_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(tuned_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_xticks(x, labels)
        ax.set_title("Test-case generator: base model vs fine-tuned Qwen3")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        plt.tight_layout()
        plt.savefig("/content/baseline_vs_finetuned.png", dpi=150)
        plt.show()

        delta = summary.loc["Fine-tuned", "overall"] - summary.loc["Base model", "overall"]
        print(f"Base overall:       {summary.loc['Base model', 'overall']:.1%}")
        print(f"Fine-tuned overall: {summary.loc['Fine-tuned', 'overall']:.1%}")
        print(f"Improvement:        {delta:+.1%}")
        '''
    ),
    markdown(
        """
        ## Step 8 — Inspect errors and save evidence

        Aggregate scores can hide serious defects. The next cell shows validation outputs that failed JSON or schema checks and writes all predictions and metrics to downloadable files.
        """
    ),
    code(
        r'''
        failures = tuned_results[(tuned_results.json_valid == 0) | (tuned_results.schema_valid == 0)]
        print(f"Fine-tuned structural failures: {len(failures)} / {len(tuned_results)}")
        for row in failures.head(3).itertuples(index=False):
            print(f"\n--- {row.example_id} ---\n{row.output[:1200]}")

        all_results.to_csv("/content/evaluation_predictions.csv", index=False)
        summary.to_csv("/content/evaluation_summary.csv")
        print("\nSaved /content/evaluation_predictions.csv")
        print("Saved /content/evaluation_summary.csv")
        print("Saved /content/training_curve.png")
        print("Saved /content/baseline_vs_finetuned.png")
        '''
    ),
    markdown(
        """
        ## Step 9 — Submission screenshot

        Run the final cell and take one screenshot that clearly shows **EXPERIMENT COMPLETE**, the comparison table, and the final scores. Add it to your Google Doc or GitHub README. Also keep the comparison chart and training curve as supporting evidence.

        A positive overall delta supports fine-tuning for this narrow, repeated task. If JSON or schema validity improves but ROUGE-L does not, the model learned the output contract while wording remained flexible. Low AC coverage is more serious because missing a requirement can let a defect escape testing.
        """
    ),
    code(
        r'''
        from IPython.display import display, Markdown

        display(Markdown("# ✅ EXPERIMENT COMPLETE"))
        display(summary.style.format("{:.1%}"))
        print(f"Held-out scenario families: {df_val.scenario_family.nunique()}")
        print(f"Held-out examples evaluated: {len(df_val)}")
        print(f"Overall improvement from fine-tuning: {delta:+.1%}")
        print("Merged model:", MERGED_DIR)
        '''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUTPUT}")
