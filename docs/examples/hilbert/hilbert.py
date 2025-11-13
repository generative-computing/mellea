import subprocess
from pathlib import Path
import os

from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.reqlib.lean import extract_lean_code, HasLeanCode, LeanCodeClearOfUnsafePrimitives, LeanCodeProvesWithoutCheating, LeanCodePreservesTheorem, LeanCodeVerifies, LeanCodeWithinLengthLimit
import mellea.stdlib.reqlib.md

class Hilbert:
    def __init__(self,
                 retriever = None,
                 reasoner: mellea.MelleaSession = None,
                 prover: mellea.MelleaSession = None,
                 lean_project_path: Path | str = None):
        self.retriever = retriever
        self.reasoner = reasoner
        self.prover = prover
        if lean_project_path is None:
            lean_project_path = os.environ.get("LEAN_PROJECT_PATH")
        self.lean_project_path = Path(lean_project_path)

        result = subprocess.run(
                ["lake", "env", "lean", "-v"],
                cwd=self.lean_project_path,
                capture_output=True,
                text=True
            )
        assert result.returncode == 0, f"Stdout: \n{result.stdout}\n Stderr: \n{result.stderr}"
        self.lean_version = result.stdout
        print("Lean version:", self.lean_version)

    def FormulateFormalStatement(self, informal_problem):
        assert isinstance(self.reasoner, mellea.MelleaSession) and isinstance(self.lean_project_path, Path)
        lean_candidate = self.reasoner.instruct(
            f"""
            Translate the following problem statement into a theorem in Lean 4 with the following instructions:
            1) Lean 4 version is {self.lean_version}
            2) There should be one theorem only and nothing else
            3) Make up a descriptive name for the theorem; make it unique and distinct from anything possibly occuring in Mathlib4.
            4) No need to write a proof; just translate the statement into a theorem and use the 'sorry' placeholder
            5) You may import Mathlib4
            6) Output in the format '```lean4\\s*\\n(.*?)```'
            The problem statement is as follows: {{informal_problem}}""",
            requirements=[
                HasLeanCode(),
                LeanCodeClearOfUnsafePrimitives(),
                f"The lean code consists of one theorem which is a true translation of the informal problem statement: {informal_problem}",
                LeanCodeVerifies(self.lean_project_path),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            user_variables={"informal_problem": informal_problem},
            return_sampling_results=True,
        )
        if lean_candidate.success:
            return extract_lean_code(str(lean_candidate.result))
        else:
            return None

    def AttemptReasonerProof(self):
        pass


    def AttemptProverLLMProof(self, theorem: str) -> str | None:
        assert isinstance(self.prover, mellea.MelleaSession) and isinstance(self.lean_project_path, Path)
        lean_candidate = self.prover.instruct(
            # f"""
            # You are given Lean 4 code that defines exactly one theorem.
            # The theorem currently ends with the placeholder 'sorry'.

            # Your task: replace the 'sorry' and give a valid Lean 4 proof for the given theorem. Here are the instructions:
            # 1) Lean 4 version is {self.lean_version}
            # 2) You may import Mathlib4. You may add import statements at the top if necessary, but you must not modify
            # anything else in the code (names, structure, or theorem statement)
            # 3) Include the given theorem statement in your output. Do not change the theorem at all.
            # 4) Reminder: module Mathlib.Tactic does not exist
            # 5) Output in the format '```lean4\\s*\\n(.*?)```'
            # The Lean 4 code is as follows: {{theorem}}""",
            f"""Think step-by-step to complete the following Lean 4 proof.
            {theorem}
            Rules:
            4. You may import Mathlib4. Do not change any of the existing imports (if any).
            5. Use proper Lean 4 syntax and conventions. Ensure the proof sketch is enclosed in
            triple backticks ```lean4```.
            6. Only include a single Lean 4 code block, corresponding to the proof along with
            the theorem statement.
            7. When dealing with large numerical quantities, avoid explicit computation as much
            as possible. Use tactics like rw to perform symbolic manipulation rather than
            numerical computation.
            8. Do NOT use sorry.
            9. Do NOT change anything in the original theorem statement.
            """,
            requirements=[
                HasLeanCode(),
                LeanCodeClearOfUnsafePrimitives(),
                LeanCodeProvesWithoutCheating(),
                # f"The lean code attempts to prove the theorem: {theorem}",
                LeanCodePreservesTheorem(theorem),
                LeanCodeVerifies(self.lean_project_path),
                LeanCodeWithinLengthLimit(30),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            user_variables={"theorem": theorem},
            return_sampling_results=True,
        )
        if lean_candidate.success:
            return extract_lean_code(str(lean_candidate.result))
        else:
            return None

    def RetrieveTheorems(problem, error_message = None):
        pass
        # 2: ▷ Theorem retrieval from Mathlib with optional parameter for error message
        # 3: if retrieval_enabled then
        # 4: search_queries ← GENERATESEARCHQUERIES(problem, error_message)
        # 5: candidate_theorems ← SEMANTICSEARCHENGINE(search_queries)
        # 6: relevant_theorems ← SELECTRELEVANTTHEOREMS(candidate_theorems, problem)
        # 7: return relevant_theorems
        # 8: else
        # 9: return ∅
        # 10: end if
        # 11: end function
