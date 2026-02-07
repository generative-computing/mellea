import subprocess
from pathlib import Path
import os
import re

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

    # SECTION 5 - HILBERT: Proof Generation

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

    def GenerateProofSketch(self, problem, relevant_theorems):
        """Generate informal proof sketch using prompts"""
        INFORMAL_PROOF_PROMPT = f"""
You are a mathematical expert whose goal is to solve problems with rigorous
mathematical reasoning.
Instructions:
1. Provide a natural language, step-by-step proof for the given problem.
2. Start from the given premises and reason step-by-step to reach the conclusion.
3. Number each step of the proof as 1, 2, and so on.
4. Be as pedantic and thorough as possible.
5. Keep each step precise, increase the number of steps if needed.
6. Do NOT gloss over any step. Make sure to be as thorough as possible.
7. Show the explicit calculations/simplifications, theorem applications and case
analysis.
8. Enclose the informal proof in <informal_proof> tags.
Problem Statement: {problem}
"""
        informal_proof = self.reasoner.instruct(
            INFORMAL_PROOF_PROMPT,
            requirements=[
                Requirement(f"The output is a valid natural language proof of the problem: {problem}."),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            user_variables={},
            return_sampling_results=True,
        )

        CREATE_LEAN_SKETCH_PROMPT = f"""
You are a Lean 4 expert who is trying to help write a proof in Lean 4.
Problem Statement:
{problem}
Relevant Theorems:
{relevant_theorems}
Informal Proof:
{informal_proof}
Instructions:
Use the informal proof to write a proof sketch for the problem in Lean 4 following
these guidelines:
- Break complex reasoning into logical sub-goals using `have` statements.
- The subgoals should build up to prove the main theorem.
- Make sure to include all the steps and calculations from the given proof in the
proof sketch.
- Each subgoal should ideally require applying just one key theorem or lemma, or a
few tactic applications.
- Base subgoals around:
- Useful theorems mentioned in the problem context
- Standard library theorems (like arithmetic properties, set operations, etc.)
- The supplied premises in the theorem statement
- Do NOT create subgoals identical to any of the given hypotheses
- Do NOT create subgoals that are more complex than the original problems. The
subgoals should be SIMPLER than the given problem.
- Do NOT skip over any steps. Do NOT make any mathematical leaps.
**Subgoal Structure Requirements:**
- **Simplicity**: Each subgoal proof should be achievable with 1-3 basic tactics
- **Atomic reasoning**: Avoid combining multiple logical steps in one subgoal
- **Clear progression**: Show logical flow: `premises → intermediate steps → final result`
- **Theorem-focused**: Design each subgoal to directly apply a specific theorem when possible
NOTE: Only add sub-goals that simplify the proof of the main goal.
When writing Lean proofs, maintain consistent indentation levels.
Rules:
1. Same proof level = same indentation: All tactics at the same logical level must
use identical indentation
2. Consistent characters: Use either tabs OR spaces consistently (don't mix)
3. Proper nesting: Indent sub-proofs one level deeper than their parent
4. Do NOT nest `have` statements in each other. Use distinct sub-goals as much as
possible. Ensure all sub goals are named. Do NOT create anonymous have statements.
5. Do NOT include any imports or open statements in your code.
6. One line = One `have` subgoal. Do NOT split subgoals across different lines.
7. Use proper Lean 4 syntax and conventions. Ensure the proof sketch is enclosed in
triple backticks ```lean```
8. Use `sorry` for all subgoal proofs - focus on structure, not implementation
9. **Do NOT use `sorry` for the main goal proof** - use your subgoals to prove it
10. NEVER use `sorry` IN the theorem statement itself
11. Ensure subgoals collectively provide everything needed for the main proof
12. Make the logical dependencies between subgoals explicit. Ensure that the subgoals
are valid and provable in Lean 4.
13. Do NOT change anything in the original theorem statement.
"""

        sketch = self.reasoner.instruct(
            CREATE_LEAN_SKETCH_PROMPT,
            requirements=[
                HasLeanCode(),
                LeanCodeClearOfUnsafePrimitives(),
                LeanCodePreservesTheorem(problem),
                LeanCodeVerifies(self.lean_project_path),
                LeanCodeWithinLengthLimit(30),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            user_variables={},
            return_sampling_results=True,
        )
        if sketch.success:
            return extract_lean_code(str(sketch.result))
        else:
            return None

    # SECTION 8 - HILBERT: Retrieval and Helper Functions

    def RetrieveTheorems(self, problem, error_message=None):
        search_queries = self.GenerateSearchQueries(problem, error_message)
        candidate_theorems = self.retriever.search_queries(search_queries, k=3)
        relevant_theorems = self.SelectRelevantTheorems(candidate_theorems, problem)
        return relevant_theorems

    def GenerateSearchQueries(self, problem, error_message=None):
        SEARCH_QUERY_PROMPT = f"""
You are helping solve a Lean theorem proving problem using the mathlib library.
Before attempting to write the proof, you must first search for relevant theorems and tactics.
Search Process:
1. Identify key concepts: Break down the problem into mathematical concepts, operations, and structures involved.
2. Generate search queries: For each concept, create informal search strings that describe:
- Relevant theorems or results (e.g., "associativity of addition", "existence of inverse elements")
- Useful tactics (e.g., "simplify arithmetic expressions", "split conjunctions")
- Properties (e.g., "group structure on integers", "metric space properties")
- Relevant definitions useful for the proof or any used theorem (e.g. "definition of a group", "definition of a
,→ metric space")
Search Query Format:
Enclose each search query in <search> tags with your informal description. Limit yourself to a maximum of 5 search
,→ queries. Make the search queries simple, concise, and clear.
Guidelines:
- You can either search by theorem name or natural language description
- Search for theorems that might automate parts of the proof
- Consider edge cases and special conditions mentioned in the problem
Problem to Solve:
{problem}
"""
        if error_message is not None:
            SEARCH_QUERY_PROMPT += f"""
You have attempted to write a proof, but obtained the following error message: {error_message}
Search for relevant theorems and tactics to resolve this.
"""

        def extract_search_queries(llm_output: str) -> list[str]:
            pattern = r"<search>(.*?)</search>"
            queries = re.findall(pattern, llm_output, flags=re.DOTALL)
            queries = [q.strip() for q in queries]
            return queries

        lean_candidate = self.prover.instruct(
            SEARCH_QUERY_PROMPT,
            requirements=[
                Requirement("The output should be in the desired format.", validation_fn=simple_validate(lambda x: bool(extract_search_queries(x)))),
                Requirement("The output should have at most 5 search queries.", validation_fn=simple_validate(lambda x: len(extract_search_queries(x)) <= 5)),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            user_variables={},
            return_sampling_results=True,
        )
        if lean_candidate.success:
            return extract_search_queries(str(lean_candidate.result))
        else:
            return None

    def SelectRelevantTheorems(self, candidate_theorems, problem):
        SEARCH_ANSWER_PROMPT = f"""
You are helping to solve a Lean theorem proving problem using the mathlib library. The problem is:
{problem}
Here are some potentially relevant theorems and definitions:
{candidate_theorems}
Instructions:
1. Select important theorems and definitions necessary to solve the problem.
2. IMPORTANT: ONLY SELECT theorems from the GIVEN list.
3. Enclose each of them in separate <theorem> tags.
4. Only state the full names of the theorems. Do NOT include the module name.
5. Select all theorems that could be useful in the intermediate steps of the proof.
"""
        def extract_theorems(llm_output: str) -> list[str]:
            pattern = r"<theorem>(.*?)</theorem>"
            queries = re.findall(pattern, llm_output, flags=re.DOTALL)
            queries = [q.strip() for q in queries]
            return queries

        lean_candidate = self.prover.instruct(
            SEARCH_ANSWER_PROMPT,
            requirements=[
                Requirement("The output should be in the desired format.", validation_fn=simple_validate(lambda x: bool(extract_theorems(x)))),
                # Requirement("The output should have at most 5 search queries.", validation_fn=simple_validate(lambda x: len(extract_search_queries(x)) <= 5)),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=5),
            user_variables={},
            return_sampling_results=True,
        )
        if lean_candidate.success:
            return extract_theorems(str(lean_candidate.result))
        else:
            return None

    def ExtractMissingIdentifiers(error_message):
        raise NotImplementedError()

    def AugmentTheorems(self, problem, error_message, existing_theorems):
        missing_ids = self.ExtractMissingIdentifiers(error_message)
        if missing_ids:
            additional_theorems = self.RetrieveTheorems(problem, error_message)
            return existing_theorems + additional_theorems
        return existing_theorems
