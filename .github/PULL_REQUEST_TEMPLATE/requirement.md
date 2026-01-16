# Requirement PR

Use this template when adding or modifying requirements in `mellea/stdlib/requirements/`.

## Description
- [ ] Link to Issue: 

<!-- Brief description of the requirement being added/modified along with an explanation for why it should be in the standard library. -->

## Implementation Checklist

### Base Class
- [ ] Extends appropriate base class:
  - `Requirement` - standard requirement
  - `ALoraRequirement` - uses specialized Intrinsic/Adapter for generation-based validation

### Validation Logic
- [ ] `validation_fn` defined (if using Python-based validation)
    - [ ] re-usable functionality within the validation_fn should be separated out into `mellea/stdlib/tools/`
- [ ] `validate` returns a `ValidationResult` with 
    - [ ] a `thunk` and `context` if using a backend to generate
    - [ ] a specific `reason` and `score` when possible

### Integration
- [ ] Requirement exported in `mellea/stdlib/requirements/__init__.py` or, if you are adding a library of requirements, from your sub-module

### Testing
- [ ] Tests added to `tests/requirements/`
- [ ] New code has 100% coverage
- [ ] Ensure existing tests and github automation passes (a maintainer will kick off the github automation when the rest of the PR is populated)
