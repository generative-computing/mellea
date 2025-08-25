from mellea import generative


# The Summarizer Library
@generative
def summarize_meeting(transcript: str) -> str:
    """Summarize the meeting transcript into a concise paragraph of main points."""


@generative
def summarize_contract(contract_text: str) -> str:
    """Produce a natural language summary of contract obligations and risks."""


@generative
def summarize_short_story(story: str) -> str:
    """Summarize a short story, with one paragraph on plot and one paragraph on broad themes."""


# The Decision Aides Library
@generative
def propose_business_decision(summary: str) -> str:
    """Given a structured summary with clear recommendations, propose a business decision."""


@generative
def generate_risk_mitigation(summary: str) -> str:
    """If the summary contains risk elements, propose mitigation strategies."""


@generative
def generate_novel_recommendations(summary: str) -> str:
    """Provide a list of novel recommendations that are similar in plot or theme to the short story summary."""


# Compose the libraries.
from typing import Literal  # noqa: E402


@generative
def has_structured_conclusion(summary: str) -> Literal["yes", "no"]:
    """Determine whether the summary contains a clearly marked conclusion or recommendation."""


@generative
def contains_actionable_risks(summary: str) -> Literal["yes", "no"]:
    """Check whether the summary contains references to business risks or exposure."""


@generative
def has_theme_and_plot(summary: str) -> Literal["yes", "no"]:
    """Check whether the summary contains both a plot and thematic elements."""


from mellea import start_session  # noqa: E402

m = start_session()
transcript = """Meeting Transcript: Market Risk Review -- Self-Sealing Stembolts Division
Date: December 1, 3125
Attendees:

Karen Rojas, VP of Product Strategy

Derek Madsen, Director of Global Procurement

Felicia Zheng, Head of Market Research

Tom Vega, CFO

Luis Tran, Engineering Liaison

Karen Rojas:
Thanks, everyone, for making time on short notice. As you've all seen, we've got three converging market risks we need to address: tariffs on micro-carburetors, increased adoption of the self-interlocking leafscrew, and, believe it or not, the "hipsterfication" of the construction industry. I need all on deck and let's not waste time. Derek, start.

Derek Madsen:
Right. As of Monday, the 25% tariff on micro-carburetors sourced from the Pan-Alpha Centauri confederacy is active. We tried to pre-purchase a three-month buffer, but after that, our unit cost rises by $1.72. That's a 9% increase in the BOM cost of our core model 440 stembolt. Unless we find alternative suppliers or pass on the cost, we're eating into our already narrow margin.

Tom Vega:
We cannot absorb that without consequences. If we pass the cost downstream, we risk losing key mid-tier OEM clients. And with the market already sniffing around leafscrew alternatives, this makes us more vulnerable.

Karen:
Lets pause there. Felicia, give us the quick-and-dirty on the leafscrew.

Felicia Zheng:
It's ugly. Sales of the self-interlocking leafscrew—particularly in modular and prefab construction—are up 38% year-over-year. It's not quite a full substitute for our self-sealing stembolts, but they are close enough in function that some contractors are making the switch. Their appeal? No micro-carburetors, lower unit complexity, and easier training for install crews. We estimate we've lost about 12% of our industrial segment to the switch in the last two quarters.

Karen:
Engineering, Luis; your take on how real that risk is?

Luis Tran:
Technically, leafscrews are not as robust under high-vibration loads. But here's the thing: most of the modular prefab sites don not need that level of tolerance. If the design spec calls for durability over 10 years, we win. But for projects looking to move fast and hit 5-year lifespans? The leafscrew wins on simplicity and cost.

Tom:
So they're eating into our low-end. That's our volume base.

Karen:
Exactly. Now let's talk about this last one: the “hipsterfication” of construction. Felicia?

Felicia:
So this is wild. We're seeing a cultural shift in boutique and residential construction—especially in markets like Beckley, West Sullivan, parts of Osborne County, where clients are requesting "authentic" manual fasteners. They want hand-sealed bolts, visible threads, even mismatched patinas. It's an aesthetic thing. Function is almost secondary. Our old manual-seal line from the 3180s? People are hunting them down on auction sites.

Tom:
Well, I'm glad I don't have to live in the big cities... nothing like this would ever happen in downt-to-earth places Brooklyn, Portland, or Austin.

Luis:
We literally got a request from a design-build firm in Keough asking if we had any bolts “pre-distressed.”

Karen:
Can we spin this?

Tom:
If we keep our vintage tooling and market it right, maybe. But that's niche. It won't offset losses in industrial and prefab.

Karen:
Not yet. But we may need to reframe it as a prestige line—low volume, high margin. Okay, action items. Derek, map alternative micro-carburetor sources. Felicia, get me a forecast on leafscrew erosion by sector. Luis, feasibility of reviving manual seal production. Tom, let's scenario-plan cost pass-through vs. feature-based differentiation.

Let's reconvene next week with hard numbers. Thanks, all."""
summary = summarize_meeting(m, transcript=transcript)

if contains_actionable_risks(m, summary=summary) == "yes":
    mitigation = generate_risk_mitigation(m, summary=summary)
    print(f"Mitigation: {mitigation}")
else:
    print("Summary does not contain actionable risks.")
if has_structured_conclusion(m, summary=summary) == "yes":
    decision = propose_business_decision(m, summary=summary)
    print(f"Decision: {decision}")
else:
    print("Summary lacks a structured conclusion.")
