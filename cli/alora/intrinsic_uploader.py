import git
from typing import Literal
import os
import tempfile
import shutil

def upload_intrinsic(
        hf_path: str,
        intrinsic_name: str,
        base_model: str,
        type: Literal["lora", "alora"],
        io_yaml: str,
):
    assert os.path.exists(io_yaml)

    # Validate and format hf_path
    if not hf_path.startswith("http") and not hf_path.startswith("git@"):
        if "/" not in hf_path:
            raise ValueError(f"Invalid hf_path format: {hf_path}")
        hf_path = f"https://huggingface.co/{hf_path}"

    temp_dir = tempfile.mkdtemp()
    try:
        # Clone the repository
        repo = git.Repo.clone_from(hf_path, temp_dir)


        # Create directory structure: intrinsic_name / base_model / adapter_type
        target_dir = os.path.join(temp_dir, intrinsic_name, base_model, type)
        is_already_intrinsic = os.path.exists(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        # Copy the io_yaml file to the target directory
        target_path = os.path.join(target_dir, os.path.basename(io_yaml))
        shutil.copy2(io_yaml, target_path)

        # Move everything else into intrinsic_name / base_model / adapter_type
        if not is_already_intrinsic:
            repo.index.move("*", target_dir)

        # Commit and push changes
        repo.index.add([target_path])
        repo.index.commit(f"Setup huggingface repo as an intrinsic.")
        repo.remotes.origin.push()
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir) 
    