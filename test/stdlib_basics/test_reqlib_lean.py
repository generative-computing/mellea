import pytest
import os

from mellea.stdlib.base import CBlock, ModelOutputThunk, Context, ChatContext
from mellea.stdlib.reqlib.lean import scan_lean_for_danger, verify_lean_code
from mellea.stdlib.requirement import default_output_to_bool


CODE_SAFE = """
def add (a b : Nat) : Nat := a + b
theorem add_comm : ∀ a b, add a b = add b a := by simp [add]
"""
CODE_UNSAFE = """
import IO
-- a comment with run_cmd
#eval IO.FS.writeFile "hello.txt" "hi"
"""
CODE_EASY = """
import Mathlib.Data.Nat.Basic

theorem one_add_one_eq_two : (1 : Nat) + 1 = 2 := by
decide
    """
CODE_HARD = """
import Mathlib.Data.Nat.Basic

theorem my_add_comm_proof (a b : Nat) : a + b = b + a := by
-- Proof by induction on `a`
induction a with
| zero =>
    -- Goal: 0 + b = b + 0
    simp
| succ a ih =>
    -- Induction hypothesis `ih`: `a + b = b + a`
    -- Goal: `(succ a) + b = b + (succ a)`

    -- Rewrite `(succ a) + b` to `succ (a + b)` using `Nat.succ_add`.
    rw [Nat.succ_add]
    -- Goal: `succ (a + b) = b + (succ a)`

    -- Rewrite `b + (succ a)` to `succ (b + a)` using `Nat.add_succ`.
    rw [Nat.add_succ]
    -- Goal: `succ (a + b) = succ (b + a)`

    -- Apply the induction hypothesis `ih : a + b = b + a`.
    -- This makes the LHS and RHS definitionally identical.
    rw [ih]
    -- At this point, the goal `succ (b + a) = succ (b + a)` is automatically
    -- closed by reflexivity. No further tactic like `rfl` is needed.
    """

def test_scan_lean_for_danger_safe():
    res_safe = scan_lean_for_danger(CODE_SAFE)
    assert res_safe["safe"] == True

def test_scan_lean_for_danger_unsage():
    res_unsafe = scan_lean_for_danger(CODE_UNSAFE)
    assert res_unsafe["safe"] == False

@pytest.mark.skipif(
    os.environ.get("LEAN_PROJECT_PATH") is None,
    reason=("LEAN_PROJECT_PATH not set. "
    "To run this test, set the environment variable:\n"
    "  export LEAN_PROJECT_PATH=/path/to/your/lean/installation\n"
    "Then run: pytest test/stdlib_basics/test_reqlib_lean.py")
)
def test_verify_lean_code_easy():
    ok, output = verify_lean_code(CODE_EASY)
    assert ok == True

@pytest.mark.skipif(
    os.environ.get("LEAN_PROJECT_PATH") is None,
    reason=("LEAN_PROJECT_PATH not set. "
    "To run this test, set the environment variable:\n"
    "  export LEAN_PROJECT_PATH=/path/to/your/lean/installation\n"
    "Then run: pytest test/stdlib_basics/test_reqlib_lean.py")
)
def test_verify_lean_code_hard():
    ok, output = verify_lean_code(CODE_HARD)
    assert ok == True

if __name__ == "__main__":
    pytest.main([__file__])
