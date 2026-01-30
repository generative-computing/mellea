"""
Example usage of the query clarification intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/query_clarification.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.context import ChatContext
from mellea.stdlib.components import Message, Document
from mellea.stdlib.components.intrinsic import rag

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Example 1: Ambiguous query that needs clarification (Positive Example)
print("=" * 80)
print("Example 1: Ambiguous query requiring clarification")
print("=" * 80)

context1 = (
    ChatContext()
    .add(
        Message(
            "user",
            "Based on the current data from MOXIE, should NASA prioritize the development of similar oxygen production technologies for future Mars missions over other resource",
        )
    )
    .add(
        Message(
            "assistant",
            "MOXIE has successfully produced oxygen from the Martian atmosphere, which could reduce the need to transport oxygen from Earth. However, the current version produces only a small fraction of what future missions would need. Whether NASA should prioritize this technology over other resources depends on various factors including mission objectives, available resources, and technological feasibility.",
        )
    )
)

query1 = "I am doing a research on women NASA astronauts. Can you give me more information about her?"
documents1 = [
    Document(
        "Former NASA Center Director, Scientist to Receive Presidential Medals - NASA\n\"Among her many accomplishments as a veteran astronaut and leader, Ellen served as the second female director of Johnson, flew in space four times, and logged nearly 1,000 hours in orbit. Jane is one of the many wizards at NASA who work every day to make the impossible, possible. The James Webb Space Telescope represents the very best of scientific discovery that will continue to unfold the secrets of our universe. We appreciate Ellen and Jane for their service to NASA, and our country.\"\nDr. Ellen Ochoa\n\nCredit: The White House\n\nOchoa retired from NASA in 2018 after more than 30 years with the agency. In addition to being an astronaut, she served a variety of positions over the years, including the 11th director of NASA Johnson, Johnson deputy center director, and director of Flight Crew Operations.\nShe joined the agency in 1988 as a research engineer at NASA's Ames Research Center in Silicon Valley, California, and moved to NASA Johnson in 1990 when she was selected as an astronaut. Ochoa became the first Hispanic woman to go to space when she served on the nine-day STS-56 mission aboard the space shuttle Discovery in 1993. She flew in space four times, including STS-66, STS-96 and STS-110.\nBorn in California, Ochoa earned a bachelor's degree in Physics from San Diego State University and a master's degree and doctorate in Electrical Engineering from Stanford University. As a research engineer at Sandia National Laboratories and NASA Ames Research Center, Ochoa investigated optical systems for performing information processing. She is a co-inventor on three patents and author of several technical papers.\n\"Wow, what an unexpected and amazing honor! I'm so grateful for all my amazing NASA colleagues who shared my career journey with me,\" said Ochoa upon hearing the news of her Presidential Medal of Freedom award.\nDuring her career, Ochoa also received NASA's highest award, the Distinguished Service Medal, and the Presidential Distinguished Rank Award for senior executives in the federal government. She has received many other awards and is especially honored to have seven schools named for her.\nOchoa also is a member of the National Academy of Engineering, and formerly chaired both the National Science Board and the Nomination Evaluation Committee for the National Medal of Technology and Innovation."
    ),
    Document(
        "Explore NASA's History - NASA\nLearn more about the African-American women who were essential to the success of early spaceflight and how their legacy continues today in NASA's commitment to a diverse workforce.\n\n \n\nFormer Astronauts\n\nGet biographies of NASA's former astronauts.\n\n \n\nHistoric Personnel\n\nThis timeline of NASA's Administrators, Deputy Administrators, and Center Directors, provides biographies of NASA's key personnel since 1958.\n\n \n\nCelebrating Asian Americans, Native Hawaiians, and Pacific Islanders in NASA's History\n\n More Stories \n\nImage Article1 Min ReadComputer Programmer and Mathematician Josephine JueArticle2 Min ReadDr. Kamlesh Lulla, Senior ScientistImage Article1 Min ReadEllison Onizuka: First Asian American in SpaceArticle5 Min Read\"Work Hard and Work Smart\": The NASA Career of Vickie Wang\n1 Min ReadVance Oyama, Searching for Life in Our Solar SystemImage Article\n1 Min ReadKalpana ChawlaImage Article\n6 Min ReadHawaii's role in NASA's space exploration programsArticle\n2 Min ReadFlight Director Pooja JesraniImage Article \n\n \n\nAstronomy and Astrophysics History\n\n Search NASA Missions \n\nHubble Space Telescope HistoryLaunched into Earth orbit in 1990, the Hubble Space Telescope has been visited by astronauts four times to make repairs and install new instruments.Learn About Hubble's HistoryNancy Grace RomanKnown as \"The Mother of Hubble,\" Nancy Grace Roman was NASA's first chief astronomer. Get to Know Nancy Grace RomanThe History of Gamma-Ray Burst ScienceOn June 1, 1973, astronomers around the world were introduced to a powerful and perplexing new phenomenon called gamma ray bursts (GRBs).Learn How GRBs Have Been Studied Since \n\nThe NACA\nNASA's precursor, the National Advisory Committee for Aeronautics (NACA) was established by Congress in 1915 to advance the United States' standing in flight. Over its 40-some years in existence, it pushed the frontiers of air technology through groundbreaking aeronautics research, preparing the country for its venturing steps into space.\n\n Learn More About the NACA about The NACA\n\n \n\nAeronautics History"
    ),
]

print(f"\nUser question: {query1}")
print(f"Number of documents retrieved: {len(documents1)}")

clarification1 = rag.clarify_query(query1, documents1, context1, backend)
print(f"\nClarification: {clarification1}")
print(
    "Expected: There are many astronauts you could be referring to, such as Ellen Ochoa, who was the first Hispanic woman to go to space and has received numerous awards, including the Presidential Medal of Freedom. Which one are you talking about?"
)

# Example 2: Clear query that doesn't need clarification (Negative Example)
print("\n" + "=" * 80)
print("Example 2: Clear query not requiring clarification")
print("=" * 80)

context2 = (
    ChatContext()
    .add(Message("user", "I have a question about my bill. What should I do?"))
    .add(
        Message(
            "assistant",
            "If you have a complaint about your bill, you can contact the California Public Utilities Commission's Consumer Affairs Branch. You can reach them by email at consumer-affairs@cpuc.ca.gov, by phone at 1-800-649-7570, or by mail at California Public Utilities Commission, Consumer Affairs Branch, 505 Van Ness Avenue, San Francisco, CA 94102. They are available Monday-Friday, 8:30 a.m.-4:30 p.m. You can also file a complaint online through their website at <https://cims.cpuc.ca.gov/complaints>.",
        )
    )
)

query2 = "and if it is about service?"
documents2 = [
    Document(
        "You can contact us by phone, email, or reporting online. You can use any one of the following contacts, depending on what you are reporting. Please be ready to give enough information to support your claims of supposed improper or unlawful behavior. If this information is not given to us, we will not be able to help you in your case.\nIf the situation poses an immediate safety threat, please call 911!\nUnsafe Conditions:\nPhone: 1-800-649-7570\nOn-Line: https://ia.cpuc.ca.gov/whblow \nFraud (deceptive practices causing financial or other losses to consumers)\nPhone: 1-800-649-7570\nE-Mail: fraudhotline@cpuc.ca.gov\nBilling, Service, and Other Complaints\nPhone: 1-800-649-7570\nOn-Line: https://cims.cpuc.ca.gov/complaints"
    ),
    Document(
        "We also cannot award claims for damages or help you determine a utility's alleged negligence or liability.  If you cannot resolve this type of problem with the utility directly, you can file a claim in civil court.\n\nIf you do not want to file your complaint online, you can send us a written complaint letter.  Be sure to include:\n\nYour name \n\nThe name the account is billed under (if it is different than your name)\n\nYour mailing address\n\nThe service address (if it is different than your mailing address)\n\nThe name of the utility or company"
    ),
]

print(f"\nUser question: {query2}")
print(f"Number of documents retrieved: {len(documents2)}")

clarification2 = rag.clarify_query(query2, documents2, context2, backend)
print(f"\nClarification: {clarification2}")
print("Expected: CLEAR")

print("\n" + "=" * 80)
