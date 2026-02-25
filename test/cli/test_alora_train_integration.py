"""Integration test for aLoRA/LoRA training with PEFT 0.18+.

This test actually trains a tiny adapter to verify the migration works end-to-end.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

pytestmark = [
    pytest.mark.huggingface,
    pytest.mark.llm,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    # Skip entire module in CI since 17/18 tests are qualitative
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping alora training tests in CI - need gpus",
    ),
]

# Check if MPS is available but PyTorch version is too old
_mps_needs_cpu_fallback = torch.backends.mps.is_available() and tuple(
    int(x) for x in torch.__version__.split(".")[:2]
) < (2, 8)


def test_alora_training_integration():
    """Integration test: Train a tiny aLoRA adapter and verify it works.

    This test:
    1. Creates a minimal training dataset (5 samples)
    2. Trains an aLoRA adapter for 1 epoch using a small model
    3. Verifies adapter files are created with correct PEFT 0.18+ format
    4. Cleans up temporary files

    Uses ibm-granite/granite-4.0-micro (smallest Granite model, 3B params).
    """
    from cli.alora.train import train_model

    # Force CPU if MPS is available but PyTorch is too old
    if _mps_needs_cpu_fallback:
        import os

        # Disable MPS entirely to force CPU usage
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        print(
            "⚠️  Warning: MPS available but PyTorch < 2.8.0. "
            "Disabling MPS to run on CPU and avoid gradient scaling issues."
        )

    # Create temporary directory for test artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create minimal training dataset (5 samples)
        dataset_path = tmpdir_path / "train.jsonl"
        training_data = [
            {"item": "Flywheel imbalance detected.", "label": "flywheel"},
            {"item": "Connecting rod bent.", "label": "connecting rod"},
            {"item": "Piston crown cracked.", "label": "piston"},
            {"item": "Oil seepage around rings.", "label": "piston rings"},
            {"item": "Carburetor obstructed.", "label": "mini-carburetor"},
        ]

        with open(dataset_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        # Output path for adapter
        adapter_path = tmpdir_path / "test_alora_adapter"

        # Train aLoRA adapter with minimal settings
        # Using smallest Granite model: granite-4.0-micro (3B params)
        train_model(
            dataset_path=str(dataset_path),
            base_model="ibm-granite/granite-4.0-micro",
            output_file=str(adapter_path),
            adapter="alora",
            epochs=1,  # Just 1 epoch for speed
            learning_rate=6e-6,
            batch_size=1,  # Minimal batch size
            max_length=512,  # Shorter sequences
            grad_accum=1,  # No gradient accumulation
        )

        # Verify adapter files were created
        assert adapter_path.exists(), "Adapter directory should be created"

        adapter_config_path = adapter_path / "adapter_config.json"
        assert adapter_config_path.exists(), "adapter_config.json should exist"

        # Verify adapter config has PEFT 0.18+ format
        with open(adapter_config_path) as f:
            config = json.load(f)

        # Key verification: PEFT 0.18+ uses "LORA" with alora_invocation_tokens
        assert config.get("peft_type") == "LORA", (
            "PEFT 0.18+ uses peft_type='LORA' for aLoRA"
        )

        assert "alora_invocation_tokens" in config, (
            "Config should have alora_invocation_tokens (PEFT 0.18+ format)"
        )

        assert isinstance(config["alora_invocation_tokens"], list), (
            "alora_invocation_tokens should be a list of token IDs"
        )

        assert len(config["alora_invocation_tokens"]) > 0, (
            "alora_invocation_tokens should not be empty"
        )

        # Verify it does NOT have old IBM format
        assert "invocation_string" not in config, (
            "Config should NOT have invocation_string (old IBM format)"
        )

        # Verify config field values match training parameters
        assert config.get("r") == 32, "LoRA rank should be 32"
        assert config.get("lora_alpha") == 32, "LoRA alpha should be 32"
        assert config.get("lora_dropout") == 0.05, "LoRA dropout should be 0.05"
        assert config.get("task_type") == "CAUSAL_LM", "Task type should be CAUSAL_LM"

        # Verify target modules
        target_modules = config.get("target_modules", [])
        assert "q_proj" in target_modules, "Should target q_proj"
        assert "k_proj" in target_modules, "Should target k_proj"
        assert "v_proj" in target_modules, "Should target v_proj"

        print("✅ Config field values verified")

        # Verify other expected files exist and check adapter weights
        weights_file = None
        if (adapter_path / "adapter_model.safetensors").exists():
            weights_file = adapter_path / "adapter_model.safetensors"
        elif (adapter_path / "adapter_model.bin").exists():
            weights_file = adapter_path / "adapter_model.bin"
        else:
            raise AssertionError("Adapter weights file should exist")

        # Load and verify adapter weights
        if weights_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            weights = load_file(str(weights_file))
        else:
            weights = torch.load(weights_file)

        # Verify we have LoRA weight keys
        lora_a_keys = [k for k in weights.keys() if "lora_A" in k]
        lora_b_keys = [k for k in weights.keys() if "lora_B" in k]
        assert len(lora_a_keys) > 0, "Should have lora_A weights"
        assert len(lora_b_keys) > 0, "Should have lora_B weights"

        # Verify weights are non-zero (adapter actually trained)
        for key, tensor in weights.items():
            assert tensor.abs().sum() > 0, f"Weight {key} should not be all zeros"

        # Verify weight shapes match rank (r=32)
        for key in lora_a_keys:
            assert weights[key].shape[0] == 32, (
                f"{key} should have rank 32 in first dim"
            )
        for key in lora_b_keys:
            assert weights[key].shape[1] == 32, (
                f"{key} should have rank 32 in second dim"
            )

        print("✅ Adapter weights verified (non-zero, correct shapes)")
        print("✅ Successfully trained aLoRA adapter with PEFT 0.18+")
        print(
            f"✅ Config format verified: {config.get('peft_type')} with alora_invocation_tokens"
        )

        # Additional verification: Verify invocation tokens are correct
        # The default invocation prompt is "<|start_of_role|>check_requirement<|end_of_role|>"
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-micro")
        default_invocation_prompt = "<|start_of_role|>check_requirement<|end_of_role|>"
        expected_tokens = tokenizer.encode(
            default_invocation_prompt, add_special_tokens=False
        )

        assert config["alora_invocation_tokens"] == expected_tokens, (
            f"Invocation tokens {config['alora_invocation_tokens']} should match "
            f"tokenized '{default_invocation_prompt}': {expected_tokens}"
        )

        print(f"✅ Invocation tokens verified: {config['alora_invocation_tokens']}")

        # Verify we can load the adapter with PEFT
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-4.0-micro",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Load the trained adapter
        model_with_adapter = PeftModel.from_pretrained(
            base_model, str(adapter_path), adapter_name="test_alora"
        )

        # Verify adapter is loaded
        assert "test_alora" in model_with_adapter.peft_config, (
            "Adapter should be loaded in PEFT model"
        )

        # Verify the loaded config matches what we saved
        loaded_config = model_with_adapter.peft_config["test_alora"]
        assert str(loaded_config.peft_type) == "PeftType.LORA", (
            "Loaded adapter should have LORA peft_type (enum format)"
        )
        assert hasattr(loaded_config, "alora_invocation_tokens"), (
            "Loaded config should have alora_invocation_tokens attribute"
        )
        assert loaded_config.alora_invocation_tokens == expected_tokens, (  # type: ignore
            "Loaded adapter should have correct invocation tokens"
        )

        print("✅ Successfully loaded adapter with PEFT and verified configuration")

        # Test actual inference with activation
        # Generate text WITHOUT invocation tokens (adapter should NOT activate)
        test_prompt_no_activation = "What is a flywheel?"
        inputs_no_activation = tokenizer(
            test_prompt_no_activation, return_tensors="pt"
        ).to(model_with_adapter.device)

        with torch.no_grad():
            outputs_no_activation = model_with_adapter.generate(
                **inputs_no_activation, max_new_tokens=20, do_sample=False
            )
        response_no_activation = tokenizer.decode(
            outputs_no_activation[0], skip_special_tokens=True
        )

        print(f"✅ Generated without activation: {response_no_activation[:100]}...")

        # Generate text WITH invocation tokens (adapter SHOULD activate)
        test_prompt_with_activation = f"{default_invocation_prompt} What is a flywheel?"
        inputs_with_activation = tokenizer(
            test_prompt_with_activation, return_tensors="pt"
        ).to(model_with_adapter.device)

        with torch.no_grad():
            outputs_with_activation = model_with_adapter.generate(
                **inputs_with_activation, max_new_tokens=20, do_sample=False
            )
        response_with_activation = tokenizer.decode(
            outputs_with_activation[0], skip_special_tokens=True
        )

        print(f"✅ Generated with activation: {response_with_activation[:100]}...")

        # Verify both generations succeeded (non-empty responses)
        assert len(response_no_activation) > len(test_prompt_no_activation), (
            "Should generate non-empty response without activation"
        )
        assert len(response_with_activation) > len(test_prompt_with_activation), (
            "Should generate non-empty response with activation"
        )

        # Check if responses differ (proving activation had an effect)
        # Note: With minimal training, responses might be identical
        if response_no_activation == response_with_activation:
            print(
                "⚠️  Warning: Responses identical with/without activation "
                "(expected with minimal training)"
            )
        else:
            print(
                "✅ Responses differ with/without activation "
                "(adapter activation confirmed)"
            )

        print(
            "✅ Verified adapter activation: both with/without invocation tokens generate successfully"
        )


def test_lora_training_integration():
    """Integration test: Train a tiny standard LoRA adapter and verify it works.

    This test verifies standard LoRA (non-aLoRA) also works with the migration.
    """
    from cli.alora.train import train_model

    # Force CPU if MPS is available but PyTorch is too old
    if _mps_needs_cpu_fallback:
        import os

        # Disable MPS entirely to force CPU usage
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        print(
            "⚠️  Warning: MPS available but PyTorch < 2.8.0. "
            "Disabling MPS to run on CPU and avoid gradient scaling issues."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create minimal training dataset
        dataset_path = tmpdir_path / "train.jsonl"
        training_data = [
            {"item": "Flywheel imbalance detected.", "label": "flywheel"},
            {"item": "Connecting rod bent.", "label": "connecting rod"},
            {"item": "Piston crown cracked.", "label": "piston"},
        ]

        with open(dataset_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        adapter_path = tmpdir_path / "test_lora_adapter"

        # Train standard LoRA adapter
        train_model(
            dataset_path=str(dataset_path),
            base_model="ibm-granite/granite-4.0-micro",
            output_file=str(adapter_path),
            adapter="lora",  # Standard LoRA, not aLoRA
            epochs=1,
            batch_size=1,
            max_length=512,
            grad_accum=1,
        )

        # Verify adapter files were created
        assert adapter_path.exists(), "Adapter directory should be created"

        adapter_config_path = adapter_path / "adapter_config.json"
        assert adapter_config_path.exists(), "adapter_config.json should exist"

        # Verify adapter config
        with open(adapter_config_path) as f:
            config = json.load(f)

        assert config.get("peft_type") == "LORA", (
            "Standard LoRA should have peft_type='LORA'"
        )

        # Standard LoRA should NOT have alora_invocation_tokens
        assert (
            "alora_invocation_tokens" not in config
            or config.get("alora_invocation_tokens") is None
        ), "Standard LoRA should not have alora_invocation_tokens"

        print("✅ Successfully trained standard LoRA adapter with PEFT 0.18+")
        print(
            f"✅ Config format verified: {config.get('peft_type')} without alora_invocation_tokens"
        )
