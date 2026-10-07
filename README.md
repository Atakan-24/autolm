# AutoLM

An experimental decoder-only transformer and evaluation pipeline for generating n8n workflow skeletons from natural-language instructions. Python and PyTorch; model, tokenizer and training code are implemented in this repository.

## Summary

- Decoder-only transformer with causal attention, residual connections and configurable model size.
- Custom BPE/tokenization code and sequential, memory-mapped training data.
- A compact workflow representation rendered into n8n JSON, with template-group split checks and a nearest-neighbor baseline.
- Evaluation of JSON, known node types, graph connections and n8n CLI import. Import is not workflow execution or semantic correctness.
- CPU-trained workflow experiment with 6,966,784 parameters. Larger GPU pretraining remains open; this is not a deployed general-purpose language model.

## Architecture

```text
workflow templates -> group splits -> mutations + instructions
                   -> tokenization -> MiniGPT training -> candidate skeletons
                                                        -> JSON/type/graph checks
                                                        -> isolated n8n import
```

`kern/` contains attention, BPE, training and checkpointing. `workflow/` contains the representation, dataset generation, baseline and sampling. `bewertung/` contains validation, import checks and saved evaluations. `schritte/` keeps explanatory model-building examples.

The compact representation reduces repetitive JSON generation, but omits workflow parameters. Two checkpoint slots retain checksums for verified fallback. Missing/corrupt metadata fails explicitly rather than silently restarting; only checkpoints produced in a trusted local directory may be loaded (`torch.load` includes optimizer/RNG state).

## Saved results and their scope

These are versioned September 12, 2026 artifacts, not freshly repeated training runs:

| Evaluation | Result | What was checked |
|---|---|---|
| [98 held-out template-derived tasks](bewertung/ergebnisse/stufe4-v3-test-schritt2400.json) | 35.7% valid first sample; 70.4% with up to 4 candidates | JSON, node types and graph checks; no n8n import in this run |
| [15 instruction tasks, single-sample import evaluation](bewertung/ergebnisse/stufe4-v3-15instr-tor4-k1-2026-09-12.json) | 7/15 passed all four checks | Structural validity and n8n import |
| [Same 15 tasks, up to 8 candidates](bewertung/ergebnisse/stufe4-v3-15instr-tor4-k8-2026-09-12.json) | 15/15 passed all four checks | Candidate selection using the validator, then import |
| [Nearest-neighbor baseline, 98 tasks](bewertung/ergebnisse/stufe4-baseline-nn-v3-2026-09-12.json) | Mean node-type Jaccard 0.523; model 0.410 | Template overlap; baseline validity is 100% by construction |

Candidate selection improves structural validity, not necessarily task fidelity. The model does not outperform the baseline on node-type overlap. The 15-task set is small; 15/15 has a 95% Wilson interval of about 79.6–100%, even before considering selection and representativeness. Old bootstrap intervals such as [100%, 100%] in the development log are superseded.

## Tests

```sh
python -m venv .venv
# activate the environment for your platform
python -m pip install -r requirements-test.txt
python -m pytest -q
```

CPU tests cover slow/fast attention and tokenizer equivalence, kill-and-resume, exact continuation of a small optimizer/RNG run, checkpoint corruption, invalid workflow shapes, graph checks and uncertainty intervals. Tests requiring the external template database are skipped when it is unavailable. CI runs the CPU suite; it does not run GPU training or n8n import.

For sampling, use `python -m workflow.demo --help`. Trained weights and the template database are not shipped. n8n import evaluation needs a separate disposable database and a locally installed CLI; never point it at a production instance. Tokenizer pickle files and checkpoints must be trusted; do not load arbitrary downloads.

## Current limits

Generated skeletons omit credentials and node parameters, and are not executed by the evaluator. Structural validity can coexist with the wrong trigger, nodes or connections. Larger GPU training, constrained decoding and broader independent evaluations remain unfinished. Updated validators may reject some historically accepted candidates; saved artifacts remain unchanged for comparison.

[Detailed development log, experiments and reproduction commands](docs/DEVELOPMENT.md)
