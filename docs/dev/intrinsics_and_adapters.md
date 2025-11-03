# Intrinsics and Adapters

## Basics
In Mellea, intrinsics are a type of Component that signals one or more of the following to a backend:
- a special adapter must be used for generation
- the input/output for generation must be transformed in a particular way
- the model options must be modified in a particular way

These changes only happen when the intrinsic is the "action" of the request. Intrinsics should usually not be used as an item in the context of generation (in fact, by default, Intrinsics have no string representation).

These changes are specified by the Adapter that corresponds to a given Intrinsic. Matching happens based on the adapter name and type.

## Parts of an Intrinsic
Intrinsics specify:
- an adapter name (ie requirement_check)
- types of adapters suitable to be used (ie alora)
- any kwargs necessary (ie a requirement like "make sure the last user message is...")

## Parts of an Adapter
Adapters specify:
- compatible backends
- adapter type
- functions for getting a path to load them

## Using Intrinsics
Mellea Intrinsics currently utilize the granite-common package for loading adapters and formatting input/outputs (https://github.com/ibm-granite/granite-common). This means Mellea only allows intrinsics/adapters that follow this pattern.

## Needed Future Work
### Custom Adapters / Intrinsics
Mellea should support custom intrinsic / adapter implementations. To do this:
- make backend `_generate_from_intrinsic` functions generic and utilize only common adapter functions
- adapters must specify a transformation function that encapsulates the input/output modifications necessary for their generation requests

### Concurrency Checks
Some backends (currently only LocalHFBackend) that allow adapters to be loaded, cannot independently utilize these adapters without impacting other generation requests.

These backends should support a generation lock that ensures requests are only performed when the correct set of adapters (or no adapters) are active.
