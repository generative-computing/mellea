import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm.auto import tqdm
import math
import os

def precompute_and_save_embeddings(
    dataset_name: str,
    text_column: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    cache_path: str = "mathlib_embeddings.pt",
    batch_size: int = 64
):
    """
    Computes and saves embeddings for a large dataset without running out of RAM
    by using a memory-mapped NumPy array on disk.
    """
    print("--- Starting Memory-Safe Embedding Pre-computation ---")

    # 1. Setup device and model
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()

    # 2. Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    texts = dataset["train"][text_column]
    total_texts = len(texts)
    print(f"Found {total_texts} texts to encode with dimension {embedding_dim}.")

    # 3. Create a memory-mapped file to store embeddings on disk
    # This avoids loading everything into RAM.
    temp_memmap_path = "embeddings.mmap"
    if os.path.exists(temp_memmap_path):
        os.remove(temp_memmap_path) # Clean up previous runs

    # Create the memory-mapped file in write mode ('w+')
    # This file acts like a NumPy array but its data is on the hard drive.
    memmap_embeddings = np.memmap(
        temp_memmap_path,
        dtype='float32',
        mode='w+',
        shape=(total_texts, embedding_dim)
    )
    print(f"Created memory-mapped file at: {temp_memmap_path}")

    # 4. Process in batches, writing directly to the memory-mapped file
    print(f"Processing in batches of size {batch_size}...")
    for i in tqdm(range(0, total_texts, batch_size), desc="Encoding Batches"):
        batch_texts = texts[i:i+batch_size]

        # Encode the batch
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device
        )

        # Write the computed embeddings directly to the correct slice in the on-disk file.
        # .cpu().numpy() is necessary to convert from a Torch tensor (on GPU/MPS) to a NumPy array (on CPU).
        start_idx = i
        end_idx = i + len(batch_texts)
        memmap_embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()

    # Ensure all data is written to disk
    memmap_embeddings.flush()

    # 5. Load the full memmap from disk and save as a final PyTorch tensor
    print("Converting memory-mapped file to a PyTorch tensor...")
    # We can now safely load the entire array from disk since it's the final step
    final_embeddings_tensor = torch.from_numpy(memmap_embeddings)

    print(f"Final embeddings tensor shape: {final_embeddings_tensor.shape}")
    print(f"Saving final embeddings to: {cache_path}")
    torch.save(final_embeddings_tensor, cache_path)

    # 6. Clean up the temporary memory-mapped file
    # The 'del' is good practice to release the file handle before deleting.
    del memmap_embeddings
    os.remove(temp_memmap_path)
    print(f"Cleaned up temporary file: {temp_memmap_path}")

    print("--- Pre-computation finished successfully! ---")


if __name__ == "__main__":
    precompute_and_save_embeddings(
        dataset_name="FrenzyMath/mathlib_informal_v4.19.0",
        text_column="informal_description",
        cache_path="mathlib_embeddings.pt",
        batch_size=128 # You can tune this. Higher is faster if you have enough RAM/VRAM.
    )
