# export PYTHONPATH="/path/to/your/dir:$PYTHONPATH"
# export LEAN_PROJECT_PATH="/path/to/your/dir"

import mellea
from docs.examples.hilbert.hilbert import Hilbert
from docs.examples.hilbert.mathlib_retriever import Retriever

retriever = Retriever()
reasoner = mellea.start_session("ollama", "gpt-oss:120b-cloud")
prover = mellea.start_session("ollama", "gpt-oss:120b-cloud")
# mellea.start_session("hf", "deepseek-ai/DeepSeek-Prover-V2-7B")
# mellea.start_session("ollama", "deepseek-v3.1:671b-cloud")
# mellea.start_session("hf", "deepseek-ai/DeepSeek-Prover-V2-671B")

hilbert = Hilbert(retriever, reasoner, prover, lean_project_path=None)
problem = hilbert.FormulateFormalStatement(
    # "1+1=2"
    # "a^2 is non-negative for all reals a",
    "x^2+x+1 is positive for all reals x",
)
print(problem)
# proof = hilbert.AttemptProverLLMProof(theorem)
# print(proof)
relevant_theorems = hilbert.RetrieveTheorems(problem)
print(relevant_theorems)
proof_sketch = hilbert.GenerateProofSketch(problem, relevant_theorems)
print(proof_sketch)
