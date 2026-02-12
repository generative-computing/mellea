"""Unit tests for aLoRA/LoRA training configuration."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from peft import LoraConfig


@pytest.mark.huggingface
def test_alora_config_creation():
    """Test that aLoRA config is created correctly with PEFT 0.18+."""
    from cli.alora.train import train_model

    # Mock all the heavy dependencies
    with (
        patch("cli.alora.train.AutoTokenizer") as mock_tokenizer_class,
        patch("cli.alora.train.AutoModelForCausalLM") as mock_model_class,
        patch("cli.alora.train.Dataset"),
        patch("cli.alora.train.SafeSaveTrainer") as mock_trainer,
        patch("cli.alora.train.get_peft_model") as mock_get_peft_model,
        patch("cli.alora.train.load_dataset_from_json") as mock_load_dataset,
        patch("cli.alora.train.DataCollatorForCompletionOnlyLM"),
    ):
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [123, 456, 789]  # Mock token IDs
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        # Mock dataset
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.__len__ = Mock(return_value=10)
        mock_load_dataset.return_value = mock_ds

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # Call train_model with aLoRA adapter
        train_model(
            dataset_path="test.jsonl",
            base_model="test-model",
            output_file="./test_output/adapter",
            adapter="alora",
            epochs=1,
        )

        # Verify get_peft_model was called
        assert mock_get_peft_model.called, "get_peft_model should be called"

        # Get the LoraConfig that was passed to get_peft_model
        call_args = mock_get_peft_model.call_args
        assert call_args is not None, (
            "get_peft_model should have been called with arguments"
        )

        peft_config = call_args[0][1]  # Second argument is the config

        # Verify it's a LoraConfig
        assert isinstance(peft_config, LoraConfig), "Should use LoraConfig"

        # Verify aLoRA-specific parameter is set
        assert hasattr(peft_config, "alora_invocation_tokens"), (
            "Config should have alora_invocation_tokens attribute"
        )
        assert peft_config.alora_invocation_tokens == [123, 456, 789], (
            "alora_invocation_tokens should match tokenized invocation prompt"
        )

        # Verify other LoRA parameters
        assert peft_config.r == 32, "Rank should be 32 for aLoRA"
        assert peft_config.lora_alpha == 32, "Alpha should be 32"
        assert peft_config.task_type == "CAUSAL_LM", "Task type should be CAUSAL_LM"


@pytest.mark.huggingface
def test_lora_config_creation():
    """Test that standard LoRA config is created correctly."""
    from cli.alora.train import train_model

    # Mock all the heavy dependencies
    with (
        patch("cli.alora.train.AutoTokenizer") as mock_tokenizer_class,
        patch("cli.alora.train.AutoModelForCausalLM") as mock_model_class,
        patch("cli.alora.train.Dataset"),
        patch("cli.alora.train.SafeSaveTrainer") as mock_trainer,
        patch("cli.alora.train.get_peft_model") as mock_get_peft_model,
        patch("cli.alora.train.load_dataset_from_json") as mock_load_dataset,
        patch("cli.alora.train.DataCollatorForCompletionOnlyLM"),
    ):
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        # Mock dataset
        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.__len__ = Mock(return_value=10)
        mock_load_dataset.return_value = mock_ds

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # Call train_model with standard LoRA adapter
        train_model(
            dataset_path="test.jsonl",
            base_model="test-model",
            output_file="./test_output/adapter",
            adapter="lora",  # Standard LoRA, not aLoRA
            epochs=1,
        )

        # Verify get_peft_model was called
        assert mock_get_peft_model.called, "get_peft_model should be called"

        # Get the LoraConfig that was passed to get_peft_model
        call_args = mock_get_peft_model.call_args
        assert call_args is not None, (
            "get_peft_model should have been called with arguments"
        )

        peft_config = call_args[0][1]  # Second argument is the config

        # Verify it's a LoraConfig
        assert isinstance(peft_config, LoraConfig), "Should use LoraConfig"

        # Verify aLoRA-specific parameter is NOT set for standard LoRA
        assert (
            not hasattr(peft_config, "alora_invocation_tokens")
            or peft_config.alora_invocation_tokens is None
        ), "Standard LoRA should not have alora_invocation_tokens"

        # Verify other LoRA parameters
        assert peft_config.r == 6, "Rank should be 6 for standard LoRA"
        assert peft_config.lora_alpha == 32, "Alpha should be 32"
        assert peft_config.task_type == "CAUSAL_LM", "Task type should be CAUSAL_LM"


@pytest.mark.huggingface
def test_invocation_prompt_tokenization():
    """Test that invocation prompt is correctly tokenized for aLoRA."""
    from cli.alora.train import train_model

    with (
        patch("cli.alora.train.AutoTokenizer") as mock_tokenizer_class,
        patch("cli.alora.train.AutoModelForCausalLM") as mock_model_class,
        patch("cli.alora.train.get_peft_model") as mock_get_peft_model,
        patch("cli.alora.train.load_dataset_from_json") as mock_load_dataset,
        patch("cli.alora.train.SafeSaveTrainer"),
        patch("cli.alora.train.DataCollatorForCompletionOnlyLM"),
        patch("cli.alora.train.os.makedirs"),
    ):
        # Setup tokenizer mock
        mock_tokenizer = Mock()
        custom_tokens = [111, 222, 333, 444]
        mock_tokenizer.encode.return_value = custom_tokens
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Setup other mocks
        mock_model_class.from_pretrained.return_value = Mock()
        mock_get_peft_model.return_value = Mock()

        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.__len__ = Mock(return_value=10)
        mock_load_dataset.return_value = mock_ds

        # Call with custom invocation prompt
        train_model(
            dataset_path="test.jsonl",
            base_model="test-model",
            output_file="./test_output/adapter",
            adapter="alora",
            epochs=1,
        )

        # Verify tokenizer.encode was called with the invocation prompt
        assert mock_tokenizer.encode.called, "Tokenizer encode should be called"

        # Verify the config has the correct tokens
        peft_config = mock_get_peft_model.call_args[0][1]
        assert peft_config.alora_invocation_tokens == custom_tokens, (
            "Config should have the tokenized invocation prompt"
        )


def test_imports_work():
    """Test that PEFT imports work correctly (no IBM alora dependency)."""
    # This test verifies the migration was successful
    from peft import LoraConfig, get_peft_model

    # Verify we can create a LoraConfig with alora_invocation_tokens
    config = LoraConfig(
        r=32, lora_alpha=32, task_type="CAUSAL_LM", alora_invocation_tokens=[1, 2, 3]
    )

    assert config.alora_invocation_tokens == [1, 2, 3], (
        "LoraConfig should support alora_invocation_tokens parameter"
    )

    # Verify get_peft_model is available
    assert callable(get_peft_model), "get_peft_model should be callable"
