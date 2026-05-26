# Component PR Checklist

Use this checklist when adding or modifying components in `mellea/stdlib/components/`.

### Protocol Compliance
- [ ] `parts()` returns list of constituent parts (Components or CBlocks)
- [ ] `format_for_llm()` returns TemplateRepresentation or string
- [ ] `_parse(computed: ModelOutputThunk)` parses model output correctly into the specified Component return type

### Content Blocks
- [ ] CBlock used appropriately for text content
- [ ] ImageBlock used for image content (if applicable)

### Integration
- [ ] Component exported in `mellea/stdlib/components/__init__.py` or, if you are adding a library of components, from your sub-module
