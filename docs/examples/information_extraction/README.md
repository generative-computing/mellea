# Information Extraction Examples

This directory contains examples for extracting structured information from unstructured text using Mellea.

## Files

### 101_with_gen_slots.py
Basic information extraction using generative slots to extract person names from text.

**Key Features:**
- Using `@generative` decorator for extraction tasks
- Type-safe extraction with `list[str]` return type
- Simple, declarative approach to information extraction
- Example with NYTimes article text

### advanced_with_m_instruct.py
More advanced extraction patterns using `m.instruct()` with structured outputs.

## Concepts Demonstrated

- **Named Entity Recognition**: Extracting person names, locations, etc.
- **Structured Extraction**: Getting typed, structured data from text
- **Type Safety**: Using Python types to constrain extraction format
- **Declarative Extraction**: Describing what to extract in docstrings

## Basic Usage

```python
from mellea import generative, start_session

@generative
def extract_all_person_names(doc: str) -> list[str]:
    """
    Given a document, extract names of ALL mentioned persons.
    Return these names as list of strings.
    """

m = start_session()
names = extract_all_person_names(m, doc=article_text)
print(names)  # ['President Obama', 'Angela Merkel']
```

## Use Cases

- **Document Processing**: Extract key information from documents
- **Data Mining**: Pull structured data from unstructured sources
- **Content Analysis**: Identify entities, relationships, and facts
- **Metadata Generation**: Create structured metadata from text

## Related Documentation

- See `generative_slots/` for more on the `@generative` decorator
- See `mellea/stdlib/components/genslot.py` for implementation details