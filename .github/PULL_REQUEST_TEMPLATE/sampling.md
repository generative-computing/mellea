# Sampling Strategy PR Checklist

Use this checklist when adding or modifying sampling strategies in `mellea/stdlib/sampling/`.

### Base Class
- [ ] Extends appropriate base class:
  - `BaseSamplingStrategy` if your changes are mostly modifying the `repair` and/or `select_from_failure` functions
  - `SamplingStrategy` if your changes involve a new `sample` method
  - Other defined sampling strategies if your implementation is similar to existing implementations

### Return Value
- [ ] Returns a properly typed `SamplingResult`. Specifically, this means:
  - `ModelOutputThunk`s in `sample_generations` are properly typed from the Component and the `parsed_repr` is the expected type.

### Integration
- [ ] Strategy exported in `mellea/stdlib/sampling/__init__.py`
