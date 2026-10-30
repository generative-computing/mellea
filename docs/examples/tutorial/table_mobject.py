# pytest: ollama, e2e
"""Working with table data structures.

Prerequisites:
    uv pip install pandas
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/tutorial/table_mobject.py
"""

from io import StringIO

import pandas

import mellea
from mellea.stdlib.components.mify import MifiedProtocol, mify


@mify(fields_include={"table"}, template="{{ table }}")
class MyCompanyDatabase:
    table: str = """| Store      | Sales   |
                    | ---------- | ------- |
                    | Northeast  | $250    |
                    | Southeast  | $80     |
                    | Midwest    | $420    |"""

    def transpose(self):
        pandas.read_csv(
            StringIO(self.table),
            sep="|",
            skipinitialspace=True,
            header=0,
            index_col=False,
        )


m = mellea.start_session()
db = MyCompanyDatabase()
assert isinstance(db, MifiedProtocol)
answer = m.query(db, "What were sales for the Northeast branch this month?")
print(str(answer))
