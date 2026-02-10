import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import git
from huggingface_hub import HfApi, HfFolder, RepoUrl, create_repo, upload_folder


def upload_intrinsic(
    weight_path: str,
    model_name: str,
    base_model: str,
    type: Literal["lora", "alora"],
    io_yaml: str,
    private: bool = True,
):
    try:
        assert os.path.exists(weight_path)
        assert os.path.exists(io_yaml)
        assert private, "not implemented."

        token = HfFolder.get_token()
        if token is None:
            raise OSError(
                "Hugging Face token not found. Run `huggingface-cli login` first."
            )

        _url: RepoUrl = create_repo(
            repo_id=model_name, token=token, private=private, exist_ok=True
        )
        hf_path = _url.url
        print(hf_path)

        temp_dir = tempfile.mkdtemp()

        # repo = git.Repo.clone_from(hf_path, temp_dir)

        # use granite-3.3-2b-instruct if the base model is granite-3.3-2b-instruct
        # use granite-3.3-2b-instruct if the base model is ibm-granite/granite-3.3-2b-instruct
        assert len(base_model.split("/")) <= 2
        base_model_path = (
            base_model if "/" not in base_model else base_model.split("/")[1]
        )

        # Create directory structure: intrinsic_name / base_model_path / adapter_type
        target_dir = os.path.join(temp_dir, model_name, base_model_path, type)
        os.makedirs(target_dir, exist_ok=True)

        # Copy the io_yaml file to the target directory
        shutil.copy2(io_yaml, weight_path)

        # Copy the model files to the target directory.
        shutil.copytree(weight_path, target_dir, dirs_exist_ok=True)

        # Commit and push changes
        upload_folder(
            repo_id=HfApi().whoami()["name"] + "/" + model_name,
            folder_path=target_dir,
            path_in_repo=os.path.join(model_name, base_model_path, type),
            commit_message="Upload adapter weights as intrinsic.",
            token=token,
        )
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    path = "/Users/nathan/dev/mellea/docs/examples/aLora/test.ckpt"
    io = "/Users/nathan/dev/mellea/docs/examples/aLora/io.yaml"
    upload_intrinsic(
        weight_path=path,
        base_model="granite-4.0-micro",
        model_name="stembolts",
        type="alora",
        io_yaml=io,
    )
