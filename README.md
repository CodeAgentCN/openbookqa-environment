# openbookqa

### Overview
- **Environment ID**: `openbookqa`
- **Short description**: OpenBookQA evaluator for science question answering.
- **Tags**: science, reasoning, nlp, multiple-choice

### Datasets
- **Primary dataset(s)**: OpenBookQA
- **Source links**: [Huggingface](https://huggingface.co/datasets/openbookqa)
- **Split sizes**: 
    - train: 4,957
    - validation: 500
    - test: 500

### Task
- **Type**: single-turn
- **Parser**: OpenBookQAParser
- **Rubric overview**: exact match on option letter (A, B, C, or D)

### Quickstart
```bash
uv run vf-eval openbookqa
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Split to evaluate |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Binary correct (1) or incorrect (0) |
| `exact_match` | Exact match on A/B/C/D |
