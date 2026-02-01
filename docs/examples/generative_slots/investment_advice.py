# pytest: ollama, llm

from typing import Literal

from mellea import MelleaSession, generative, start_session


@generative
def analyze_client_profile(profile: str) -> dict:
    """Extract risk_tolerance, time_horizon, and liquidity_needs from the profile."""


@generative
def should_have_equity_investments(client_analysis: dict) -> bool:
    """Determine if the client should have equity investments based on their risk_tolerance."""


@generative
def generate_advice_letter(profile: str) -> str:
    """Generate a personalized investment advice letter based on the client profile."""


@generative
def list_mentioned_products(text: str) -> list[str]:
    """Return a list of financial products mentioned in the text (e.g. ETFs, bonds, options)."""


@generative
def detect_prohibited_language(text: str) -> Literal["clean", "prohibited"]:
    """Detect whether the letter contains prohibited phrases like 'guaranteed returns' or 'no risk'."""


def check_semantic_preconditions(client_analysis: dict) -> None:
    required_fields = ["risk_tolerance", "time_horizon", "liquidity_needs"]
    missing = [
        f for f in required_fields if f not in client_analysis or not client_analysis[f]
    ]
    if missing:
        raise ValueError(f"Missing client data: {', '.join(missing)}")


def check_semantic_postconditions(
    m: MelleaSession,
    letter: str,
    client_analysis: dict,
    products: list[str],
    lang_flag: str,
) -> None:
    if should_have_equity_investments(m, client_analysis=client_analysis):
        if (
            "ETF" not in [p.upper() for p in products]
            and "index fund" not in [p.lower() for p in products]
            and "mutual fund" not in [p.lower() for p in products]
        ):
            raise ValueError(
                "Letter must recommend at least one diversified investment vehicle (e.g., ETF, mutual fund, or index fund)."
            )
    if client_analysis.get("risk_tolerance") == "low":
        if any(
            p.lower() in ("options", "futures", "swaps", "derivatives")
            for p in products
        ):
            raise ValueError(
                "Client has low risk tolerance — advice must not include derivatives."
            )
    if lang_flag == "prohibited":
        raise ValueError("Letter contains prohibited compliance language.")


def render_financial_advice(m: MelleaSession, profile: str) -> str:
    # Step 1: Analyze profile
    analysis = analyze_client_profile(m, profile=profile)
    check_semantic_preconditions(analysis)

    # Step 2: Generate advice
    letter = generate_advice_letter(m, profile=profile)

    # Step 3: Run postcondition validations
    products = list_mentioned_products(m, text=letter)
    lang_flag = detect_prohibited_language(m, text=letter)
    check_semantic_postconditions(m, letter, analysis, products, lang_flag)

    return letter


if __name__ == "__main__":
    m = start_session()
    profile_text = (
        "Client is a 62-year-old retiree with a conservative outlook.\n"
        "They need access to funds within 3-5 years and are concerned about volatility.\n"
        "Liquidity is important, and they've asked for low-risk options."
    )

    try:
        print(render_financial_advice(m, profile_text))
    except ValueError as e:
        print("🚫 Validation failed:", e)
