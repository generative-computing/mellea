# How KG-RAG Fits the Theme of Mellea

This document explains how the refactored KG-RAG implementation embodies Mellea's core philosophy and design patterns.

## Mellea's Core Philosophy

Mellea is designed around these principles:

1. **Composability** - Build complex systems from simple, composable parts
2. **Type Safety** - Use Pydantic models for structured, validated outputs
3. **Declarative Programming** - Declare what you want, not how to get it
4. **Context Management** - Efficient memory and KV-cache reuse
5. **Robustness** - Requirements and sampling strategies for reliable outputs
6. **Functional Patterns** - Pure functions, immutable data, clear data flow

## How KG-RAG Aligns with Each Principle

### 1. Composability ✅

**Mellea Pattern**: Components that can be composed into larger pipelines

**KG-RAG Implementation**:
```python
# KGRagComponent is a proper Mellea Component
class KGRagComponent(Component):
    async def execute(self, query: str, ...) -> str:
        # Can be composed with other components
        pass

# Use standalone
answer = await kg_rag.execute(query="...")

# Or compose in a pipeline
result = await session.instruct(
    instruction="Answer comprehensively",
    components=[kg_rag, summarizer, validator],
    user_variables={"query": "..."}
)
```

**What This Achieves**:
- KG-RAG is no longer a standalone system
- It's a reusable component in larger Mellea applications
- Can be mixed with other RAG methods, reasoning components, etc.

### 2. Type Safety ✅

**Mellea Pattern**: Pydantic models for all structured data

**KG-RAG Implementation**:
```python
# Every LLM output is a typed Pydantic model
class QuestionRoutes(BaseModel):
    reason: str
    routes: List[List[str]]

class RelevantEntities(BaseModel):
    reason: str
    relevant_entities: Dict[str, float]

# Type-safe function signatures
@generative
async def break_down_question(...) -> QuestionRoutes:
    pass

# IDE knows the return type
result: QuestionRoutes = await break_down_question(...)
result.routes  # Type-safe access with autocomplete
```

**What This Achieves**:
- Catch errors at development time, not runtime
- Self-documenting code with clear contracts
- Better IDE support (autocomplete, refactoring)
- Automatic validation of LLM outputs

### 3. Declarative Programming ✅

**Mellea Pattern**: `@generative` decorator for LLM-powered functions

**KG-RAG Implementation**:
```python
# Before: Imperative - manually construct messages, call API, parse JSON
response = await generate_response(
    self.session,
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": user_message}],
    max_tokens=2048
)
routes = maybe_load_json(response)["routes"]

# After: Declarative - describe what you want
@generative
async def break_down_question(query: str, ...) -> QuestionRoutes:
    """Break down the question into solving routes.

    Question: {query}
    """
    pass

result = await break_down_question(query="...")
```

**What This Achieves**:
- Focus on what to generate, not how
- Prompts live with the function (locality)
- Automatic JSON parsing via return type
- Less boilerplate, more signal

### 4. Context Management ✅

**Mellea Pattern**: Efficient context and KV-cache reuse

**KG-RAG Implementation**:
```python
from mellea.stdlib.base import Context

# Reuse context across related calls
ctx = Context()
ctx.add_system_message(f"You are an expert in {domain}.")

# Multiple calls reuse the same context
for entity in entities:
    result = await prune_relations(
        context=ctx,  # Reuses cached context
        entity=entity,
        ...
    )
```

**What This Achieves**:
- Reduces redundant token usage
- Faster inference with KV-cache reuse
- Lower API costs
- Better memory efficiency

### 5. Robustness ✅

**Mellea Pattern**: Requirements + Sampling Strategies

**KG-RAG Implementation**:
```python
from mellea.stdlib.requirement import Requirement
from mellea.stdlib.sampling.rejection_sampling import RejectionSamplingStrategy

# Define what makes a valid output
requirements = [
    Requirement(
        name="valid_json",
        requirement="Output must be valid JSON",
        validator=is_valid_json
    ),
    Requirement(
        name="routes_present",
        requirement="Must contain at least one route",
        validator=has_nonempty_list("routes")
    ),
]

# Automatic retry until requirements are met
result = await session.instruct(
    instruction=prompt,
    requirements=requirements,
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

**What This Achieves**:
- Replaces custom `@llm_retry` decorator
- Declarative validation constraints
- Built-in retry logic
- Composable validators
- Integrates with Mellea's inference-time scaling

### 6. Functional Patterns ✅

**Mellea Pattern**: Pure functions, immutable data, clear data flow

**KG-RAG Implementation**:
```python
# Each generative function is pure
@generative
async def extract_topic_entities(
    query: str,
    query_time: str,
    route: List[str],
    domain: str
) -> TopicEntities:
    """Extract entities from query."""
    pass

# Clear data flow through the pipeline
query_obj = Query(query="...", query_time=...)
routes = await break_down_question(query_obj)
entities = await extract_topic_entities(routes[0])
aligned = await align_topic(query_obj, entities)
answer = await reasoning(query_obj, aligned)
```

**What This Achieves**:
- Easy to test individual functions
- Clear input/output contracts
- No hidden state or side effects
- Easier to reason about and debug

## Concrete Examples of Integration

### Example 1: Multi-Route Validation with Majority Voting

**Before**: Custom validation logic
```python
# Manual consensus checking in kg_model.py:892-935
stop = False
for route in queries[2:]:
    route_results.append(await explore_one_route(route))
    if len(route_results) >= 2:
        stop, final = await self.validation(queries, attempt, route_results)
        if stop:
            break
```

**After**: Use Mellea's MajorityVotingStrategy
```python
from mellea.stdlib.sampling.majority_voting import MajorityVotingStrategy

# Explore routes in parallel and vote
strategy = MajorityVotingStrategy(
    num_samples=len(routes),
    vote_method="consensus"
)

final_answer = await session.instruct(
    instruction=f"Answer: {query}",
    strategy=strategy,
    user_variables={"entities": entities_str, "triplets": triplets_str}
)
```

### Example 2: LLM-as-Judge Evaluation

**Before**: Custom evaluation logic in `eval.py`

**After**: Use Mellea's test-based evaluation
```python
from mellea.stdlib.test_based_eval import LLMAsAJudge

judge = LLMAsAJudge(
    session=eval_session,
    rubric="""
    Evaluate if the prediction correctly answers the question.
    Score 1.0 for correct, 0.5 for partial, 0.0 for incorrect.
    """,
)

score = await judge.evaluate(
    input=query,
    output=prediction,
    reference=ground_truth,
)
```

### Example 3: Composing with Other Components

**New Capability**: KG-RAG as part of a larger system

```python
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.intrinsics.structured_output import StructuredOutput

# Build a multi-stage pipeline
kg_retrieval = KGRagComponent(...)
answer_formatter = StructuredOutput(schema=AnswerSchema)
quality_checker = RequirementChecker(requirements=[...])

# Compose them
pipeline = session.create_pipeline([
    kg_retrieval,      # Retrieve from KG
    answer_formatter,  # Format the answer
    quality_checker,   # Validate quality
])

result = await pipeline.execute(query="...")
```

## What Makes This "Mellea-Style"?

### Before (External Library Approach)
- KG-RAG feels like an external system that happens to use Mellea for API calls
- Custom infrastructure (retry logic, token counting, JSON parsing)
- Monolithic architecture
- Hard to extend or compose

### After (Integrated Mellea Approach)
- KG-RAG is a **native Mellea component**
- Uses Mellea's patterns throughout (Requirements, @generative, Components)
- Composable with other Mellea components
- Easy to extend and adapt

### The Key Difference

**External Library**:
```python
# Uses Mellea as a backend
kg_model = KGModel(session=mellea_session)
answer = await kg_model.generate_answer(query)
```

**Integrated Component**:
```python
# IS a Mellea component
kg_rag = KGRagComponent(session, domain="movie")

# Can use standalone
answer = await kg_rag.execute(query)

# Or compose with other Mellea patterns
result = await session.instruct(
    "Answer using knowledge graph and verify",
    components=[kg_rag, verifier],
    requirements=[accuracy_req],
    strategy=MajorityVotingStrategy(num_samples=3)
)
```

## Benefits of This Integration

### 1. Unified Mental Model
- Everything follows Mellea patterns
- No context switching between "Mellea code" and "KG-RAG code"
- Consistent APIs across the codebase

### 2. Better Composability
- Mix KG-RAG with vector RAG, web search, tool calling
- Build sophisticated multi-stage pipelines
- Reuse components across different applications

### 3. Reduced Code Duplication
- Don't reinvent retry logic, validation, token tracking
- Leverage Mellea's battle-tested infrastructure
- Focus on domain logic, not plumbing

### 4. Easier Onboarding
- New developers already know Mellea patterns
- Less custom infrastructure to learn
- Better documentation via types and docstrings

### 5. Future-Proof
- As Mellea evolves, KG-RAG benefits automatically
- New sampling strategies, requirements, components just work
- Easier to maintain and extend

## Comparison Table

| Aspect | Before | After |
|--------|--------|-------|
| **Architecture** | Monolithic KGModel class | Composable KGRagComponent |
| **LLM Calls** | Manual `generate_response()` | `@generative` functions |
| **Validation** | Custom `@llm_retry` | Mellea Requirements |
| **Prompts** | Python strings in code | Jinja2 templates |
| **Types** | Loose (dicts, strings) | Strong (Pydantic models) |
| **Sampling** | Manual route validation | SamplingStrategies |
| **Composition** | Standalone system | Mellea Component |
| **Testing** | Hard to test | Easy to test pure functions |
| **Extensibility** | Requires forking | Just add components |

## Conclusion

The refactored KG-RAG doesn't just **use** Mellea—it **embodies** Mellea's philosophy:

- ✅ **Composable** - Proper Component architecture
- ✅ **Type-safe** - Pydantic models throughout
- ✅ **Declarative** - @generative functions
- ✅ **Robust** - Requirements and sampling strategies
- ✅ **Functional** - Pure functions, clear data flow
- ✅ **Efficient** - Context management and caching

This transformation makes KG-RAG:
- A **showcase** of Mellea patterns
- A **template** for building similar systems
- A **component** in the Mellea ecosystem

Instead of being a separate project that happens to use Mellea, it's now a **native Mellea application** that demonstrates the framework's power and flexibility.

## Try It Yourself

```bash
# Original version
uv run --with mellea run/run_qa.py

# Refactored version
uv run --with mellea python demo_refactored.py

# Compare the code
diff kg_model.py kg_rag_refactored.py
```

See [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) for detailed migration instructions.
