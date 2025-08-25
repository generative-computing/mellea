
from typing import Any
from dataclasses import dataclass
from string import Template
from pydantic import BaseModel, create_model


def supercede(cls1: type, cls2: type) -> type:
    """
    Return the more specific class if one is a subclass of the other.
    Raise ValueError if neither is a subclass of the other.
    """
    if issubclass(cls1, cls2) and cls1 is not cls2:
        return cls1
    elif issubclass(cls2, cls1) and cls1 is not cls2:
        return cls2
    elif cls1 is cls2:
        return cls1
    else:
        raise ValueError(f"{cls1.__name__} and {cls2.__name__} are unrelated.")


class Pattern:

    def __init__(self, pattern:str, **types:type[Any]):
        # e.g.,
        self.pattern = Template(pattern)
        self.types   = types


    def defaulted_types(self) -> dict[str, type[Any]]:
        # defaults the variable type to str.
        # we do not do this defaulting in the constructor because
        # composition may add in additional type information.
        # in other words, type defaults are performed in the very last step.

        return {
            v : (
                self.types[v]
                if v in self.types
                else str
            )
            for variable in self.pattern.get_identifiers()
        }


    def model(self) -> type[BaseModel]:

        # dynamically create a new Pydantic model class

        return create_model("PatterbModel", **self.defaulted_types())  # type: ignore


    def format(self) -> str:

        return (
            "Fill in the blank in the following template, " +
            "where the placeholders are denoted as $var for a variable named 'var'. " +
            "Answer in a json schema shown after the template. " +
            "\nTemplate:\n" +
            self.pattern.template
            "\nSchema:\n" +
            self.model().dump_json_schema()
        )


    def __and__(self, other) -> 'Pattern':
        # WIP. not a final state.
        # We should actually keep the original Pattern instance, rather than immediately merging them,
        # because we must perform query optimization later.

        types = self.types.copy()
        for v in other.types.keys():
            if v in types:
                types[v] = supersede(other.types[v], types[v])
            else:
                types[v] = other.types[v]

        return Pattern(
            self.pattern.template + " " + other.pattern.template,
            **types)


if __name__ == "__main__":

    from mellea import start_session, SimpleContext
    m = start_session()
    p = Pattern("the height of $mountain is $height meters.",
                mountain=str,
                height=int)

    print(m.instruct(p.format(),
                     format=p.model()))

