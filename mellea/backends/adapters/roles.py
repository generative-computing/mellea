"""Advisory registry of known adapter roles.

:data:`KNOWN_ROLES` is a frozenset of role strings that Mellea ships with. It is
advisory only: callers are warned (not rejected) when a role outside this set is used.
"""

KNOWN_ROLES: frozenset[str] = frozenset(
    {
        # Core intrinsics
        "context-attribution",
        "requirement-check",
        "requirement_check",
        "uncertainty",
        # RAG intrinsics
        "answerability",
        "citations",
        "context_relevance",
        "hallucination_detection",
        "query_clarification",
        "query_rewrite",
        # Guardian intrinsics
        "policy-guardrails",
        "guardian-core",
        "factuality-detection",
        "factuality-correction",
    }
)
