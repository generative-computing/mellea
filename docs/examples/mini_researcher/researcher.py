from collections.abc import Callable
from functools import cache

from openai import BaseModel
from pydantic import ValidationError

from docs.examples.helper import w
from docs.examples.mini_researcher import RAGDocument
from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Requirement, SamplingResult
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# #############################
# Helper functions
# #############################


@cache
def get_session():
    """Get M session (change model here)."""
    return MelleaSession(backend=OllamaModelBackend(model_ids.IBM_GRANITE_4_MICRO_3B))


@cache
def get_guardian_session():
    """Get M session for the guardian model."""
    return MelleaSession(
        backend=OllamaModelBackend(model_ids.IBM_GRANITE_GUARDIAN_3_0_2B)
    )


def is_a_true_subset_of_b(a: list[str], b: list[str]) -> bool:
    """Check if a is true subset of b."""
    all_in = True
    for e in a:
        if e not in b:
            all_in = False
            break
    return all_in


def create_check_word_count(max_words: int) -> Callable[[str], bool]:
    """Generate a maximum-word-count validation function."""

    def cc(s: str):
        return len(s.split()) <= max_words

    return cc


# ########################################
# Functions for each step of the pipeline
# ########################################


def step_is_input_safe(guardian_session: MelleaSession, docs: list[str]) -> bool:
    """Check if the list of docs has no harm."""
    is_safe = True
    for i_doc, doc in enumerate(docs):
        print(f"\nChecking Doc {i_doc + 1}/{len(docs)}", end="...")
        inspect = guardian_session.chat(doc)
        if str(inspect).upper().startswith("YES"):
            is_safe = False
            print("FAILED")
            break
        else:
            print("OK")
    return is_safe


def step_summarize_docs(
    s: MelleaSession, docs: list[str], user_args: dict
) -> list[str]:
    """Generate a task-specific document summary for each doc."""
    summaries = []
    for i_doc, doc in enumerate(docs):  # type: ignore
        print(f"\nSummarizing doc {i_doc + 1}/{len(docs)}", end="...")
        summary = s.instruct(
            f"Summarize the following document to answer the question: ' How {{current_subtopic}} impacts {{main_topic}}?' \n Document: {doc}",
            requirements=["Use maximal 3 sentences."],
            user_variables=user_args,
        )
        summaries.append(str(summary))
        print("done.")
    return summaries


def step_generate_outline(
    s: MelleaSession, user_args: dict, context: list[RAGDocument]
) -> list[str]:
    """Generate a report outline using constraint decoding (formatted output)."""
    print("\n Generating outline", end="...")

    class SectionTitles(BaseModel):
        section_titles: list[str]

    def must_have_sections(out: str) -> bool:
        try:
            stt = SectionTitles.model_validate_json(out)
            return is_a_true_subset_of_b(
                ["Introduction", "Conclusion", "References"], stt.section_titles
            )
        except ValidationError:
            return False

    def max_sub_sections(out: str) -> bool:
        try:
            stt = SectionTitles.model_validate_json(out)
            return (
                len(stt.section_titles) <= 3 + user_args["max_subsections"]
            )  # x + Intro, Conclusion, Ref
        except ValidationError:
            return False

    ## Define Requirements
    req_outline = Requirement(
        "In addition to main body of the report, the report should also include these standard sections:  Introduction,  Conclusion, and References",
        validation_fn=simple_validate(must_have_sections),
    )

    req_num_sections = Requirement(
        f"Limit the number of subsections to a maximum of {user_args['max_subsections']}.",
        validation_fn=simple_validate(max_sub_sections),
    )

    outline_context = {
        f"Document {i + 1}": f"## Title: {d.title}, ## Source: {d.source}"
        for i, d in enumerate(context)
    }

    ## Generate
    outline_result = s.instruct(
        description="Create an outline for a report on how {{current_subtopic}} impacts {{main_topic}}. Use the Context Documents provided as guideline for the sections.",
        # output_prefix="# Introduction",
        requirements=[req_outline, req_num_sections],
        user_variables=user_args,
        grounding_context=outline_context,
        strategy=RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results=True,
        format=SectionTitles,
    )

    st = SectionTitles.model_validate_json(outline_result.value)

    if isinstance(outline_result, SamplingResult):
        if not outline_result.success:
            for i_r, r in enumerate(outline_result.sample_generations):
                if r == outline_result.result:
                    print("Validation Results:")
                    for v in outline_result.sample_validations[i_r]:
                        print(f"\t{v[1]} <- {v[0].description}")

    print("done.")
    return st.section_titles


def step_write_full_report(
    m: MelleaSession,
    max_words: int,
    user_args: dict,
    summaries: list[str],
    outline: list[str],
) -> str:
    """Merge summaries and outline into a single report."""
    print("\nWriting full report", end="...")

    ## Define Requirements
    req_focus = Requirement("Stay focused on the topic, avoid unrelated information.")
    req_language = Requirement(f"Write the report in {user_args['language']} language.")
    req_tone = Requirement("Use an {{tone}} tone throughout the report.")
    req_length = Requirement(
        f"The report should have a maximum length of {max_words} words.",
        validation_fn=simple_validate(create_check_word_count(max_words=max_words)),
    )

    user_args.update(
        {
            "context": "\n".join(summaries),
            "outline": "\n".join([f"* {o}" for o in outline]),
        }
    )

    ## Generate
    report_result = m.instruct(
        description="Context:\n{{context}}\nSummarize the relevant information available into a detailed report on how {{current_subtopic}} impacts {{main_topic}}.\n\nFollow this outline:\n{{outline}}",
        requirements=[req_focus, req_length, req_language, req_tone],
        user_variables=user_args,
        strategy=RejectionSamplingStrategy(loop_budget=2, requirements=[req_length]),
        return_sampling_results=True,
    )

    if isinstance(report_result, SamplingResult):
        if not report_result.success:
            for i_r, r in enumerate(report_result.sample_generations):
                if r == report_result.result:
                    print("Validation Results:")
                    for v in report_result.sample_validations[i_r]:
                        print(f"\t{v[1]} <- {v[0].description}")

    print("done.")
    return report_result.value


def research_subtopic(main_topic: str, subtopic: str, context: list[RAGDocument]):
    """Start MiniResearcher here."""
    user_args = {
        "context_docs": context,
        "current_subtopic": subtopic,
        "main_topic": main_topic,
        "max_subsections": 5,
        "existing_headers": "",
        "relevant_written_contents": "",
        "date": "April 26, 2025",
        "language": "English",
        "total_words": 1000,
        "tone": "professional",
    }

    m = get_session()
    guardian_session = get_guardian_session()

    # Step 0: check for Harm in input
    safe_input = step_is_input_safe(guardian_session, docs=[c.content for c in context])
    if not safe_input:
        return {"error": "Input not safe"}

    # Step 1: Summarize each doc
    summaries = step_summarize_docs(
        m, docs=[c.content for c in context], user_args=user_args
    )
    summaries_str = "\n\n".join([w(s) for s in summaries])
    print(f"Summaries: \n{summaries_str}")

    # Step 2: Generate Outline
    outline = step_generate_outline(m, user_args=user_args, context=context)
    outline_str = "\n".join(outline)
    print(f"Outline:\n{outline_str}")

    # Step 3: Merge all for the final report
    full_report = step_write_full_report(
        m,
        user_args=user_args,
        max_words=user_args["total_words"],
        summaries=summaries,
        outline=outline,
    )
    return full_report


if __name__ == "__main__":
    # data from an external file
    from docs.examples.mini_researcher.context_docs import documents as ctx_docs

    report = research_subtopic(
        "IBM earnings outlook", "Recent IBM acquisitions", context=ctx_docs
    )

    print(f"\nFull Report:\n\n{w(report)}")
