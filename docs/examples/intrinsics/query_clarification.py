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
            "the most common form of user interface used on pc's today is called a",
        )
    )
    .add(
        Message(
            "assistant",
            "The most common form of user interface used on personal computers today is the graphical user interface (GUI). It typically follows the windows, icons, menus, pointer (WIMP) paradigm, which includes visual elements like windows, icons, menus, and a pointer to interact with information stored in computers.",
        )
    )
    .add(Message("user", "Advantages of using command line interface"))
    .add(
        Message(
            "assistant",
            "While I don't have specific data comparing it directly to the most common user interfaces today (which are GUIs), the command line interface (CLI) does have several advantages as described in the documents:\n1. **Efficiency**: A CLI can be faster for entering large vocabularies of commands or queries, especially when coupled with a wide range of options. This is often the case with operating system command shells.\n2. **Low resource requirements**: CLIs generally require fewer system resources to implement compared to GUIs.\n3. **Ease of accessing options**: With commands and their options given in a few characters, an experienced user might find it easier to access and remember the options in a CLI.\n4. **Automation of repetitive tasks**: CLIs often support mechanisms for storing frequently used command sequences in a disk file for re-use. This can extend to a scripting language that takes parameters and variable options.\n5. **Command history**: A command-line history can be kept, allowing review or repetition of commands.\nHowever, it's important to note that these advantages can depend on the specific use case and the user's familiarity with the CLI.",
        )
    )
)

query1 = "Is the language that descended from Common Brittonic still spoken today, or has it become extinct?"
documents1 = [
    Document(
        "Common Brittonic developed from the Insular branch of the Proto - Celtic language that developed in the British Isles after arriving from the continent in the 7th century BC . The language eventually began to diverge ; some linguists have grouped subsequent developments as Western and Southwestern Brittonic languages . Western Brittonic developed into Welsh in Wales and the Cumbric language in the Hen Ogledd or `` Old North '' of Britain ( modern northern England and southern Scotland ) , while the Southwestern dialect became Cornish in Cornwall and South West England and Breton in Armorica . Pictish is now generally accepted to descend from Common Brittonic , rather than being a separate Celtic language . Welsh and Breton survive today ; Cumbric and Pictish became extinct in the 12th century . Cornish had become extinct by the 19th century but has been the subject of language revitalization since the 20th century ."
    ),
    Document(
        "Wales , Cornwall and Brittany continued to retain a distinct Brittonic culture , identity and language , which they have maintained to the present day . The Welsh language and Breton language remain widely spoken , and the Cornish language , once close to extinction , has experienced a revival since the 20th century . The vast majority of place names and names of geographical features in Wales , Cornwall and Brittany are Brittonic , and Brittonic family and personal names remain common ."
    ),
    Document(
        "The Britons spoke an Insular Celtic language known as Common Brittonic . Brittonic was spoken throughout the island of Britain ( in modern terms , England , Wales and Scotland ) , as well as offshore islands such as the Isle of Man , Scilly Isles , Orkneys , Hebrides , Isle of Wight and Shetlands . According to early medieval historical tradition , such as The Dream of Macsen Wledig , the post-Roman Celtic - speakers of Armorica were colonists from Britain , resulting in the Breton language , a language related to Welsh and identical to Cornish in the early period and still used today . Thus the area today is called Brittany ( Br . Breizh , Fr . Bretagne , derived from Britannia ) ."
    ),
]

print(f"\nUser question: {query1}")
print(f"Number of documents retrieved: {len(documents1)}")

clarification1 = rag.clarify_query(query1, documents1, context1, backend)
print(f"\nClarification: {clarification1}")
print(
    f"Expected: There are many languages that descended from Common Brittonic, such as: 1) Welsh, which is still spoken today, particularly in Wales, 2) Breton, which is still spoken in Brittany, and 3) Cornish, which had become extinct but has since been revitalized. Which one are you referring to?"
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
