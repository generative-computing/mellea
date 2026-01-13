# Component PR

Use this template when adding or modifying components in `mellea/stdlib/components/`.

## Description
- [ ] Link to Issue: 

<!-- Brief description of the component being added/modified along with an explanation for why it should be in the standard library. -->

## Implementation Checklist

### Protocol Compliance
- [ ] `parts()` returns list of constituent parts (Components or CBlocks)
- [ ] `format_for_llm()` returns TemplateRepresentation or string
- [ ] `_parse(computed: ModelOutputThunk)` parses model output correctly into the specified Component return type

### Content Blocks
- [ ] CBlock used appropriately for text content
- [ ] ImageBlock used for image content (if applicable)

### Integration
- [ ] Component exported in `mellea/stdlib/components/__init__.py` or, if you are adding a library of components, from your sub-module

### Testing
- [ ] Tests added to `tests/components/`
- [ ] New code has 100% coverage
- [ ] Ensure existing tests and github automation passes
