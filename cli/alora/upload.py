import os

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder


def upload_model(weight_path: str, model_name: str, private: bool = True):
    """Upload a trained adapter (LoRA/aLoRA) to Hugging Face Hub.

    Args:
        weight_path (str): Directory containing adapter weights (from save_pretrained).
        model_name (str): Target model repo name (e.g., "acme/carbchecker-alora").
        private (bool): Whether the repo should be private. Default: True.

    Requires:
        - `HF_TOKEN` set in environment or via `huggingface-cli login`.
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Adapter directory not found: {weight_path}")

    # Create repo if not exists
    token = HfFolder.get_token()
    if token is None:
        raise OSError(
            "Hugging Face token not found. Run `huggingface-cli login` first."
        )

    try:
        create_repo(repo_id=model_name, token=token, private=private, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create or access repo {model_name}: {e}")

    print(
        f"Uploading adapter from '{weight_path}' to 'https://huggingface.co/{model_name}' ..."
    )

    upload_folder(
        repo_id=model_name,
        folder_path=weight_path,
        path_in_repo=".",  # Root of repo
        commit_message="Upload adapter weights",
        token=token,
    )
