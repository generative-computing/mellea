from mellea.stdlib.components.docs.richdocument import TableQuery
from mellea.stdlib.components.mify import MifiedProtocol, mify
from mellea.stdlib.session import start_session


# Mify works on python objects and classes. Apply it to your own
# custom class or object to start working with mellea.
@mify
class MyCustomerClass:
    def __init__(self, name: str, last_purchase: str) -> None:
        self.name = name
        self.last_purchase = last_purchase


# Now when you instantiate an object of that class, it will also
# have the fields and members necessary for working with mellea.
c = MyCustomerClass("Jack", "Beans")
assert isinstance(c, MifiedProtocol)


# You can also mify objects ad hoc.
class MyStoreClass:
    def __init__(self, purchases: list[str]) -> None:
        self.purchases: list[str]


store = MyStoreClass(["Beans", "Soil", "Watering Can"])
mify(store)
assert isinstance(store, MifiedProtocol)

# Now, you can use these objects in MelleaSessions.
store.format_for_llm()
m = start_session()
m.act(store)


# However, unless your object/class has a __str__ function,
# this won't do much good by itself. You need to specify how
# mellea should process these objects as text. You can do this by
# parameterizing mify.
@mify(stringify_func=lambda x: f"Chain Location: {x.location}")  # type: ignore
class MyChain:
    def __init__(self, location: str):
        self.location = location


# M operations will now utilize that string representation of the
# object when interacting with it.
m.query(MyChain("Northeast"), "Where is my chain located?")


# For more complicated representations, you can utilize mify
# to interact with our templating system. Here, we know that a
# TableQuery calls its underlying object's to_markdown function.
# Since our class has the same process, we can use that template.
# We can also specify that our class should use either a template with it's own
# class name or the Table template when not querying.
@mify(query_type=TableQuery, template_order=["*", "Table"])
class MyCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""

    def to_markdown(self):
        return self.table


# Mellea also allows you to specify the fields you want to
# include from your class and a corresponding template that
# takes those fields.
@mify(fields_include={"table"}, template="{{ table }}")
class MyOtherCompanyDatabase:
    table: str = """| Store      | Sales   |
| ---------- | ------- |
| Northeast  | $250    |
| Southeast  | $80     |
| Midwest    | $420    |"""


m.query(
    MyOtherCompanyDatabase(), "What were sales for the Northeast branch this month?"
)


# By default, mifying and object will also provide any functions
# of your class/object to models as tools in m functions that support tools.
# The default behavior only includes functions that have docstrings without
# [no-index] in it.
@mify(funcs_include={"from_markdown"})
class MyDocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "MyDocumentLoader":
        doc = MyDocumentLoader()
        # Your parsing functions here.
        doc.content = text
        return doc


# m.transform will be able to call the from_markdown function to return
# the poem as a MyDocumentLoader object.
m.transform(MyDocumentLoader(), "Write a poem.")
