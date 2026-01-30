# pytest: ollama, qualitative, llm, requires_heavy_ram

from mellea.backends import model_ids
from mellea.backends.model_ids import IBM_GRANITE_3_3_8B
from mellea.stdlib.components.docs.richdocument import RichDocument

rd = RichDocument.from_document_file("https://arxiv.org/pdf/1906.04043")

from mellea.stdlib.components.docs.richdocument import Table  # noqa: E402

table1: Table = rd.get_tables()[0]
print(table1.to_markdown())

from mellea import start_session  # noqa: E402
from mellea.backends import ModelOption  # noqa: E402

m = start_session(model_id=model_ids.META_LLAMA_3_2_3B)
for seed in [x * 12 for x in range(5)]:
    table2 = m.transform(
        table1,
        "Add a column 'Model' that extracts which model was used in Feature description or 'None' if none.",
        model_options={ModelOption.SEED: seed},
    )
    if isinstance(table2, Table):
        print(table2.to_markdown())
        break
    else:
        print("==== TRYING AGAIN after non-useful output.====")
