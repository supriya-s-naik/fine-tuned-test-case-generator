# Week 5 Project — Fine-Tuned Test Case Generator

**Name:** [Your name]  
**Date:** [Submission date]  
**Repository:** [GitHub repository URL]  
**Loom video:** [Loom URL, if submitting the custom project route]

## Problem and use case

QA teams spend substantial time translating user stories and acceptance criteria into repeatable test cases. A general-purpose model can draft tests, but the response may omit acceptance criteria, use inconsistent fields, or produce prose that a test-management system cannot parse. I fine-tuned Qwen3 1.7B to generate a consistent JSON schema with explicit acceptance-criterion traceability and positive, negative, and boundary tests.

## Dataset

I used 80 synthetic labelled examples spanning multiple software domains. Sixty examples from 12 scenario families were used for training. Twenty examples from four complete scenario families were held out for validation. Holding out whole families reduced leakage from closely related template variants.

Each target response contains `id`, `title`, `type`, `preconditions`, `steps`, `expected_result`, and `covers`. The `covers` field links test cases to identifiers such as `AC1`, which makes requirements coverage measurable.

## Training approach

I trained a LoRA adapter on the conversational `Qwen/Qwen3-1.7B` checkpoint using LLaMA Factory. The evaluated run used a learning rate of `1e-5`, two epochs, batch size 1, gradient accumulation 4, LoRA rank 8, a 1024-token cutoff, and FP16 compute on a T4 GPU. LoRA kept the base weights frozen and trained only small low-rank parameter updates. I then merged the adapter to produce a standalone inference model. An initial run on the pretraining-only Base checkpoint collapsed during autoregressive generation, so I switched to the conversational checkpoint and reduced the learning rate.

## Evaluation

The unchanged base model and merged fine-tuned model used the same system prompt, deterministic decoding, and 20 held-out examples. I measured JSON validity, schema validity, acceptance-criteria coverage, test-type coverage, and ROUGE-L reference similarity.

| Metric | Base model | Fine-tuned model | Delta |
|---|---:|---:|---:|
| JSON validity | 100% | 100% | 0 pp |
| Schema validity | 80% | 75% | -5 pp |
| AC coverage | 100% | 100% | 0 pp |
| Test-type coverage | 50% | 52% | +2 pp |
| ROUGE-L | 51% | 52% | +1 pp |
| Overall mean | 76.2% | 75.7% | -0.5 pp |

## Result and interpretation

The fine-tuned model preserved perfect JSON validity and acceptance-criteria coverage while slightly improving test-type coverage and ROUGE-L. Strict schema validity fell by five percentage points, producing an overall change of -0.5 percentage points. All five tuned structural failures came from the held-out appointment-booking family: the model returned `expected_result` as a nonempty array rather than the required string. The content remained useful, but it violated the specified contract.

This result does not support deploying the adapter instead of the conversational base model. The baseline was already strong, and the small synthetic dataset did not add measurable overall value. The next experiment should make every field type explicit, add diverse examples with multi-part expected results represented as strings, and test additional epochs against the same untouched validation set.

The most consequential metric is acceptance-criteria coverage because an omitted requirement can allow a defect to escape. JSON and schema validity determine whether the response can be integrated automatically. ROUGE-L provides useful reference comparison but does not prove semantic correctness by itself.

## Evidence

Paste the screenshot from the final notebook cell here. It should show **EXPERIMENT COMPLETE**, both result rows, the number of held-out examples, and the improvement delta.

**[INSERT SUCCESSFUL RUN SCREENSHOT HERE]**

Also include, if desired:

- training loss curve;
- base-versus-fine-tuned comparison chart;
- one held-out input and fine-tuned output.

## Limitations and next steps

This dataset is synthetic and small. A production version should use de-identified, QA-reviewed examples from the target product, expand rare and high-risk behaviors, and add human evaluation for correctness, duplication, unsupported assumptions, and risk coverage. A deployment should retain human review until performance is validated on real work and should route malformed or low-confidence outputs for correction.
