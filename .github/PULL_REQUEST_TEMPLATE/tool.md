# Tool PR

Use this template when adding or modifying components in `mellea/stdlib/tools/`.

## Description
- [ ] Link to Issue: 

<!-- Brief description of the tool being added/modified along with an explanation for why it should be in the standard library. -->

## Implementation Checklist

### Protocol Compliance
- [ ] Ensure compatibility with existing backends and providers
    - For most tools being added as functions, this means that calling `convert_function_to_tool` works

### Integration
- [ ] Tool exported in `mellea/stdlib/tools/__init__.py` or, if you are adding a library of components, from your sub-module

### Testing
- [ ] Tests added to `tests/stdlib/tools/`
- [ ] New code has 100% coverage
- [ ] Ensure existing tests and github automation passes (a maintainer will kick off the github automation when the rest of the PR is populated)
