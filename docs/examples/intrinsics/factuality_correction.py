# pytest: huggingface, e2e

"""Example usage of the factuality correction intrinsic.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/factuality_correction.py
```
"""

from mellea import start_backend, model_ids
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import guardian

user_text = "Why does February have 28 days?"
response_text = "February has 28 days because it was named after the Roman god of war, Mars, and the Romans believed that the month should have an even number of days to symbolize balance and fairness in war."
document = Document(
    "The Gregorian calendar's oldest ancestor, the first Roman calendar, had "
    "10 months instead of 12. Roman king Numa Pompilius added January and "
    "February to sync the calendar with the lunar year. He wanted to avoid even "
    "numbers due to Roman superstition and subtracted a day from each of the 30-day "
    "months to make them 29. The lunar year consists of 355 days, and at least "
    "1 month out of the 12 needed to contain an even number of days. Numa chose "
    "February, a month that would be host to Roman rituals honoring the dead, as "
    "the unlucky month to consist of 28 days. Despite changes in the calendar, "
    "February's 28-day length has stuck.\\n\\nFebruary has 28 days due to Roman "
    "superstition. The Roman calendar originally divided the year from March to "
    "December into 10 months of either 29 or 31 days, based on lunar cycles. Later, "
    "January and February were added to cover the full year. The Romans considered "
    "even numbers unlucky, so February was chosen to have 28 days as this was "
    "when they honored their dead. Despite changes to the calendar, February's "
    "unique 28-day length survived.\\n\\nThe Gregorian calendar's oldest ancestor, "
    "the first Roman calendar, had a glaring difference in structure from its "
    "later variants: it consisted of 10 months rather than 12. The Roman king "
    "Numa Pompilius added January and February to the original 10 months. "
    "However, Numa wanted to avoid having even numbers in his calendar, as Roman "
    "superstition at the time held that even numbers were unlucky. He subtracted a "
    "day from each of the 30-day months to make them 29. The lunar year consists of "
    "355 days, which meant that he now had 56 days left to work with. In the end, "
    "at least 1 month out of the 12 needed to contain an even number of days. So "
    "Numa chose February, a month that would be host to Roman rituals honoring "
    "the dead, as the unlucky month to consist of 28 days.\\n\\nThe Roman calendar "
    "was originally established by Romulus and consisted of ten months, each "
    "having 30 or 31 days. The calendar was then revised by Numa Pompilius, who "
    "divided winter between the two months of January and February, shortened "
    "most other months, and brought everything into alignment with the solar "
    "year by some system of intercalation. The calendar was lunisolar and had "
    "important days such as kalends, nones, and ides, which seem to have derived "
    "from the new moon, the first-quarter moon, and the full moon respectively. "
    "The calendar was conservatively maintained until the Late Republic, but "
    "intercalation was not always observed, causing the civil calendar to vary "
    "from the solar year. Caesar reformed the calendar in 46 BC, creating the "
    "Julian calendar, which was an entirely solar one. The Julian calendar was "
    "designed to have a single leap day every fourth year.\\n\\nThe month of "
    "February was named after the purification rituals of ancient Rome, not the "
    "Roman god of war, Mars. The name February comes from the Latin februare, "
    'meaning \\"to purify.\\"\\n\\nThe ancient Romans believed that even '
    "numbers were unlucky, which is why February has 28 days instead of 30. They "
    "preferred to have more 31-day months.\\n\\nFebruary has 28 days due to "
    "Roman superstition. The Roman calendar originally divided the year into "
    "10 months of either 29 or 31 days, based on lunar cycles. Later, January "
    "and February were added to cover the full year. The Romans considered even "
    "numbers unlucky, so February was chosen to have 28 days as this was when they "
    "honored their dead. This unique 28-day length survived the changes from "
    "the Julian to the Gregorian calendar.\\n\\nFebruary's 28 days date back to "
    "the second king of Rome, Numa Pompilius. Before he became king, Rome's "
    "lunar calendar was just 10 months long, beginning in March and ending in "
    "December. The time between December and March was considered unimportant "
    "as it had nothing to do with the harvest. When Numa Pompilius took reign, "
    "he decided to make the calendar more accurate by lining it up with the year's "
    "12 lunar cycles. He added January and February to the end of the calendar. "
    "Because Romans believed even numbers to be unlucky, each month had an odd "
    "number of days, which alternated between 29 and 31. However, to reach "
    "355 days, one month had to be an even number. February was chosen to be "
    "the unlucky month with 28 days. This choice may be due to the fact that "
    "Romans honored the dead and performed rites of purification in "
    "February.\\n\\nThe ancient Romans believed that even numbers were evil and "
    "they tried to make every month have an odd number of days. However, they "
    "couldn't do this for February, making it the only month with an even number "
    "of days. The reason behind this belief is not explained in the provided context."
)

# Create the backend.
ctx, backend = start_backend("hf", model_id=model_ids.IBM_GRANITE_4_MICRO_3B, context_type="chat")
ctx = ctx.add(document).add(Message("user", user_text)).add(Message("assistant", response_text))

result = guardian.factuality_correction(ctx, backend)
print(f"Result of factuality correction: {result}")  # corrected response string
