from stembolts_intrinsic import (
    async_stembolt_failure_analysis,
    stembolt_failure_analysis,
)

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.stdlib.context import ChatContext

if __name__ == "__main__":
    backend = LocalHFBackend(IBM_GRANITE_4_MICRO_3B)

    welcome_msg = (
        "==   Welcome to the Self-Sealing Stembolt Part Diagnostic System.   =="
    )
    print("=" * len(welcome_msg))
    print("=" * len(welcome_msg))
    print(welcome_msg)
    print("=" * len(welcome_msg))
    print("=" * len(welcome_msg))

    mechanics_notes = None
    while True:
        if mechanics_notes is None:
            mechanics_notes = "Oil seepage when stembolt is oriented diagonally, even when oil pin is tightened"
            print(f"Mechanic: {mechanics_notes}")
        else:
            mechanics_notes = input("Mechanic: ")

        result, ctx = stembolt_failure_analysis(
            notes=mechanics_notes, ctx=ChatContext(), backend=backend
        )
        print(f"Assistant: {result}")
