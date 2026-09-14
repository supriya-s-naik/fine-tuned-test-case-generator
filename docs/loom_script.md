# Loom Walkthrough Script

Target length: 3–5 minutes.

## 1. Introduce the problem (30 seconds)

“For my Week 5 custom project, I fine-tuned Qwen3 1.7B to generate software test cases from a user story and acceptance criteria. The business problem is that QA teams repeatedly perform this conversion, while general model outputs can be inconsistent or omit requirements.”

Show the example input and JSON output in the README.

## 2. Explain the dataset (45 seconds)

Open `data/user_story_test_cases.csv`.

“The dataset has 80 labelled examples across several software domains. Each answer follows a strict schema and maps test cases back to acceptance-criterion IDs. I train on 60 examples and hold out 20 examples from four complete scenario families. That group-level split makes validation harder and reduces leakage.”

Show the `split`, `user_story`, `acceptance_criteria`, and `expected_test_cases` columns.

## 3. Show the fine-tuning workflow (60 seconds)

Open the notebook around Steps 3–6.

“The notebook converts the training rows to ShareGPT format and registers them in LLaMA Factory. I use LoRA, which freezes the base model and learns small low-rank updates. After training, I plot loss, compare the original model on held-out data, and merge the adapter into a standalone Qwen checkpoint.”

Briefly show the LLaMA Board settings and the training curve.

## 4. Explain evaluation (60 seconds)

Show the comparison table and chart.

“Because this is generation, exact-match accuracy would be misleading. I measure whether the output is valid JSON, follows the schema, covers every AC ID, includes positive, negative, and boundary tests, and resembles the reference using ROUGE-L. Both models receive the same prompt and deterministic decoding settings.”

Read the base and fine-tuned overall results and delta. Mention the strongest improvement and any weak metric.

## 5. Conclude with limitations (30 seconds)

“This proves the fine-tuning workflow and tests whether the adapter learns a stable output contract. The synthetic dataset is too small for production. My next step would be to add de-identified, human-reviewed examples and evaluate semantic correctness and risk coverage with QA engineers.”

End on the notebook’s **EXPERIMENT COMPLETE** cell.
