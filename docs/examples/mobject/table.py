from io import StringIO

import pandas

import mellea
from mellea.stdlib.components.mify import mify


@mify(fields_include={"table"}, template="{{ table }}")
class MyCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""

    def __init__(self, *, table: str | None = None):
        if table is not None:
            self.table = table

    def update_sales(self, store: str, amount: str):
        """Update the sales for a specific store."""
        table_df = pandas.read_csv(
            StringIO(self.table),
            sep="|",
            skipinitialspace=True,
            header=0,
            index_col=False,
        )
        # Remove unnamed columns and columns that don't exist.
        table_df = table_df.drop(table_df.filter(regex="Unname").columns, axis=1)

        # Sometimes extra whitespace gets added to the column names and row values. Remove it.
        table_df.columns = table_df.columns.str.strip()
        table_df = table_df.map(lambda x: x.strip() if isinstance(x, str) else x)

        table_df.loc[table_df["Store"] == store, "Sales"] = amount
        self.table = table_df.to_csv(sep="|", index=False, header=True)
        return self

    def transpose(self):
        """Transpose the table."""
        return (
            pandas.read_csv(
                StringIO(self.table),
                sep="|",
                skipinitialspace=True,
                header=0,
                index_col=False,
            )
            .transpose()
            .to_csv(StringIO(), sep="|", index=False, header=True)
        )


if __name__ == "__main__":
    m = mellea.start_session()
    db = MyCompanyDatabase()
    print(m.query(db, "What were sales for the Northeast branch this month?").value)
    result = m.transform(db, "Update the northeast sales to 1250.")
    print(type(result))
    print(db.table)
    print(m.query(db, "What were sales for the Northeast branch this month?"))
    result = m.transform(db, "Transpose the table.")
    print(result)
