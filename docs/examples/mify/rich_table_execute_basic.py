# This is an example of using the RichDocument class.
import os

from mellea import start_session
from mellea.backends import model_ids
from mellea.backends import ModelOption
from mellea.core import FancyLogger
from mellea.stdlib.components.docs.richdocument import RichDocument, Table

FancyLogger.get_logger().setLevel("ERROR")

"""
Here we demonstrate the use of the (internally m-ified) class
RichDocument and Table that are wrappers around Docling documents.
More about Docling here: https://github.com/docling-project/docling
"""

# Using a paper from arxive as source file
source = "https://arxiv.org/pdf/1906.04043"


# Since processing a document can take some time, we can also cache
# the document loading (by uncommenting below)
def cached_doc():
    tmp_file = "tmp_doc.json"
    if os.path.exists(tmp_file):
        return RichDocument.load(tmp_file)
    else:
        loaded_doc = RichDocument.from_document_file(source)
        loaded_doc.save(tmp_file)
        return loaded_doc


# rd = cached_doc()

# load the document
rd = RichDocument.from_document_file(source)

# extract the first table as (m-ified) Table object
table1 = rd.get_tables()[0]
print(table1.to_markdown())

# start M session with local Llama model
m = start_session(
    model_id=model_ids.META_LLAMA_3_2_3B,
    model_options={ModelOption.MAX_NEW_TOKENS: 500},
)
print("==> Outputs:")
# apply transform on the Table and make sure that the returned object is a Table. Try up to 5 times.
for seed in [x * 12 for x in range(5)]:
    table_transformed = m.transform(
        table1,
        "Add a column 'Model' that extracts which model was used or 'None' if none.",
        model_options={ModelOption.SEED: seed},
    )
    if isinstance(table_transformed, Table):
        print(table_transformed.to_markdown())
        break
    else:
        print("==== TRYING AGAIN after non-useful output.====")
# print(f"Last prompt: {m.last_prompt()}")
