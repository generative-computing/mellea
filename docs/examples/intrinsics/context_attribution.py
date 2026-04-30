# pytest: huggingface, e2e

"""Example usage of the context attribution intrinsic.

Intrinsic function that finds sentences in prior conversation messages and RAG
documents that were most important to the LLM in generating each sentence in
the assistant response.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/context_attribution.py
```
"""

import json

from mellea import model_ids, start_backend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import core

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_MICRO_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx = (
    ctx.add(
        Message(
            "user",
            "Who were the members of The Metal Ono Band, which was formed by Yoko Ono "
            "in 1976 to explore her interest in heavy metal music?",
        )
    )
    .add(
        Message(
            "assistant",
            "I'm sorry, but I don't have the data to answer that specific question. ",
        )
    )
    .add(
        Message(
            "user", "What was the concept behind the formation of the Plastic Ono Band?"
        )
    )
)

assistant_response = (
    "The Plastic Ono Band was formed by John Lennon and Yoko Ono in 1969 as a "
    "collaborative vehicle for their artistic and personal projects. They decided to "
    "credit all their future efforts to this conceptual and collaborative group after "
    "their marriage in 1969."
)
documents = [
    Document(
        doc_id="0",
        text="The Plastic Ono Band is a band formed by John Lennon and Yoko Ono in 1969 "
        "as a vehicle for their collaborative and solo projects. Lennon and Ono had "
        "begun a personal and artistic relationship in 1968, collaborating on several "
        "experimental releases. Following their marriage in 1969, they decided that all "
        "of their future endeavours would be credited to a conceptual and collaborative "
        "vehicle, Plastic Ono Band. The band would go on to feature a rotating lineup "
        "of many musicians, including Eric Clapton, Klaus Voormann, Alan White, Billy "
        "Preston, Jim Keltner, Delaney & Bonnie and Friends, and Lennon's former "
        "Beatles bandmates George Harrison and Ringo Starr. Lennon and Ono left the UK "
        "to settle in New York City during the fall of 1971. In Greenwich Village, the "
        "couple became more politically active and began writing protest songs. These "
        "songs became the basis for their next album, Some Time in New York City. As "
        "backing, they enlisted the help of New York band Elephant's Memory, consisting "
        "of guitarist Wayne 'Tex' Gabriel, bassist Gary Van Scyoc, saxophonist Stan "
        "Bronstein, keyboardist Adam Ippolito, keyboardist John La Boosca, and drummer "
        "Richard Frank, Jr. Phil Spector produced, and Jim Keltner also played on the "
        'album. The album was released on 12 June 1972, credited to "John & '
        "Yoko/Plastic Ono Band with Elephant's Memory plus Invisible Strings\". Some "
        "Time in New York City included a second disc, entitled Live Jam, which "
        "included the recordings from the 1969 Peace for Christmas concert and the 1971 "
        "performance with Frank Zappa. Ono and Lennon continued their work with "
        "Elephant's Memory throughout 1972, performing as the Plastic Ono Elephant's "
        "Memory Band (which also included Jim Keltner). On 30 August, they performed a "
        'pair of benefit concerts at Madison Square Garden. The benefit, entitled "One '
        'to One", was organised by Geraldo Rivera to raise money for children with '
        "mental challenges. By this time, La Boosca had departed the band, and the "
        "concert saw the addition of John Ward on bass. The concert was filmed and "
        "recorded, later released in February 1986 as the album Live In New York City. "
        "They also performed at the Jerry Lewis MDA Labor Day Telethon. The last "
        "collaboration of the Plastic Ono Elephant's Memory Band was Ono's double album "
        "Approximately Infinite Universe. It was recorded throughout the fall of 1972, "
        "and was released in January 1973.",
    ),
    Document(
        doc_id="1",
        text="The Plastic Ono Band is a band formed by John Lennon and Yoko Ono in 1969 "
        "as a vehicle for their collaborative and solo projects. Lennon and Ono had "
        "begun a personal and artistic relationship in 1968, collaborating on several "
        "experimental releases. Following their marriage in 1969, they decided that all "
        "of their future endeavours would be credited to a conceptual and collaborative "
        "vehicle, Plastic Ono Band. The band would go on to feature a rotating lineup "
        "of many musicians, including Eric Clapton, Klaus Voormann, Alan White, Billy "
        "Preston, Jim Keltner, Delaney & Bonnie and Friends, and Lennon's former "
        "Beatles bandmates George Harrison and Ringo Starr. By the beginning of 1973, "
        "recording had begun in earnest on Ono's next album, Feeling the Space, "
        "featuring a new group of studio musicians. The newest incarnation of the "
        "Plastic Ono Band featured guitarist David Spinozza, keyboardist Ken Ascher, "
        "bassist Gordon Edwards, percussionists Arthur Jenkins and David Friedman, "
        "saxophonist Michael Brecker, pedal steel guitarist Sneaky Pete Kleinow, as "
        "well as regular contributor Jim Keltner. The album would be released in "
        "November. Throughout 1973, Lennon and Ono's relationship became strained. By "
        'August, the two had begun a period of separation that Lennon called "The Lost '
        'Weekend". Lennon began the recording of his own album, Mind Games, using the '
        'same players as on Feeling the Space, dubbed "The Plastic U.F.Ono Band". '
        "Around the time of the album's release in November, Lennon moved to Los "
        "Angeles with new lover May Pang. In October, Lennon began the recording of an "
        "album of rock 'n' roll oldies (a contractual obligation due to a lawsuit). "
        'These featured many Plastic Ono Band regulars (including much of the "U.F.Ono '
        'Band", Klaus Voorman, and the return of Phil Spector to the production chair), '
        "but upon release in 1975 as Rock 'n' Roll, it was credited to Lennon alone. "
        "The sessions for Rock 'n' Roll were extremely troubled, and the sessions were "
        "abandoned until a later date. In July 1974, Lennon returned to New York to "
        'record Walls and Bridges. The new "Plastic Ono Nuclear Band" featured both '
        "old and new faces, with Jim Keltner, Kenneth Ascher, and Arthur Jenkins "
        "continuing from Mind Games, the returns of Klaus Voorman, Nicky Hopkins, and "
        "Bobby Keys, and the addition of guitarists Jesse Ed Davis and Eddie Mottau. "
        "Recording was finished in August, and the album was released 26 September and "
        "4 October in the US and UK respectively. Walls and Bridges would prove to be "
        "the last release of new material by the Plastic Ono Band in the 1970s. Lennon "
        "subsequently returned to his marriage with Ono and retired from music following "
        "the birth of his son Sean. The compilation Shaved Fish was released in October "
        "1975, Lennon's last release credited to the Plastic Ono Band. Upon his and "
        "Ono's return to music in 1980 for the album Double Fantasy, they played with "
        "an all-new group of studio musicians who were not billed as any variation of "
        "the Plastic Ono Band name. Lennon was shot and killed shortly after the release "
        "of the album.",
    ),
    Document(
        doc_id="2",
        text="John Winston Ono Lennon  (9 October 1940 - 8 December 1980) was an English "
        "singer, songwriter, and peace activist who co-founded the Beatles, the most "
        "commercially successful band in the history of popular music. He and fellow "
        "member Paul McCartney formed a much-celebrated songwriting partnership. Along "
        "with George Harrison and Ringo Starr, the group would ascend to world-wide "
        "fame during the 1960s. During his marriage to Cynthia, Lennon's first son "
        "Julian was born at the same time that his commitments with the Beatles were "
        "intensifying at the height of Beatlemania. Lennon was touring with the Beatles "
        "when Julian was born on 8 April 1963. Julian's birth, like his mother "
        "Cynthia's marriage to Lennon, was kept secret because Epstein was convinced "
        "that public knowledge of such things would threaten the Beatles' commercial "
        "success. Julian recalled that as a small child in Weybridge some four years "
        'later, "I was trundled home from school and came walking up with one of my '
        "watercolour paintings. It was just a bunch of stars and this blonde girl I "
        "knew at school. And Dad said, 'What's this?' I said, 'It's Lucy in the sky "
        "with diamonds.'\" Lennon used it as the title of a Beatles song, and though it "
        "was later reported to have been derived from the initials LSD, Lennon "
        "insisted, \"It's not an acid song.\" McCartney corroborated Lennon's "
        "explanation that Julian innocently came up with the name. Lennon was distant "
        "from Julian, who felt closer to McCartney than to his father. During a car "
        "journey to visit Cynthia and Julian during Lennon's divorce, McCartney composed "
        'a song, "Hey Jules", to comfort him. It would evolve into the Beatles song '
        '"Hey Jude". Lennon later said, "That\'s his best song. It started off as a '
        "song about my son Julian ... he turned it into 'Hey Jude'. I always thought "
        "it was about me and Yoko but he said it wasn't.\" Lennon's relationship with "
        "Julian was already strained, and after Lennon and Ono moved to Manhattan in "
        "1971, Julian would not see his father again until 1973. With Pang's "
        "encouragement, arrangements were made for Julian (and his mother) to visit "
        "Lennon in Los Angeles, where they went to Disneyland. Julian started to see "
        "his father regularly, and Lennon gave him a drumming part on a Walls and "
        "Bridges track. He bought Julian a Gibson Les Paul guitar and other instruments, "
        "and encouraged his interest in music by demonstrating guitar chord techniques. "
        'Julian recalls that he and his father "got on a great deal better" during the '
        'time he spent in New York: "We had a lot of fun, laughed a lot and had a great '
        'time in general." In a Playboy interview with David Sheff shortly before his '
        'death, Lennon said, "Sean was a planned child, and therein lies the difference. '
        "I don't love Julian any less as a child. He's still my son, whether he came "
        "from a bottle of whiskey or because they didn't have pills in those days. He's "
        'here, he belongs to me, and he always will." He said he was trying to '
        "re-establish a connection with the then 17-year-old, and confidently predicted, "
        '"Julian and I will have a relationship in the future." After his death it was '
        "revealed that he had left Julian very little in his will.",
    ),
]

result = core.find_context_attributions(assistant_response, documents, ctx, backend)
print(f"Result of context attribution intrinsic:\n{json.dumps(result, indent=2)}")
