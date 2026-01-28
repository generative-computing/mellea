# aLoRA Examples

This directory contains examples demonstrating Adaptive Low-Rank Adaptation (aLoRA) for efficient constraint checking and requirement validation.

## Files

### 101_example.py
A comprehensive example showing how to use aLoRA adapters for fast constraint checking with Granite models.

**Key Features:**
- Loading and using custom aLoRA adapters for constraint checking
- Comparing validation speed with and without aLoRA
- Using `ALoraRequirement` for efficient requirement validation
- Demonstrates significant speedup when using aLoRA adapters

**Usage:**
```bash
python docs/examples/aLora/101_example.py
```

### Supporting Files

- **prompt_config.json**: Configuration for training aLoRA adapters
- **stembolt_failure_dataset.jsonl**: Training dataset for the failure mode constraint
- **checkpoints/alora_adapter/**: Pre-trained aLoRA adapter checkpoint

## Concepts Demonstrated

- **aLoRA Adapters**: Using specialized adapters for constraint checking
- **Constraint Validation**: Fast requirement checking with aLoRA
- **Performance Optimization**: Comparing validation times with/without aLoRA
- **Custom Requirements**: Creating domain-specific validation requirements
- **Backend Integration**: Adding aLoRA adapters to HuggingFace backends

## Training Your Own aLoRA

To train custom aLoRA adapters for your constraints:

```bash
m alora train --config docs/examples/aLora/prompt_config.json
```

See `cli/alora/` for more details on training aLoRA adapters.

## Related Documentation

- See `docs/dev/requirement_aLoRA_rerouting.md` for aLoRA architecture details
- See `mellea/backends/adapters/` for adapter implementation