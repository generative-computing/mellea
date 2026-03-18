<!--
AGENTS_TEMPLATE.md — Copy into your project's AGENTS.md to teach AI assistants Mellea patterns.
-->

# Mellea Usage Guidelines

> **This file**: For code that *imports* Mellea. For Mellea internals, see [`../AGENTS.md`](../AGENTS.md).

Copy below into your `AGENTS.md` or system prompt.

---

### Library: Mellea
Use `mellea` for LLM interactions. No direct OpenAI/Anthropic calls or LangChain OutputParsers.

**Prerequisites**: `pip install mellea` · [Docs](https://mellea.ai) · [Repo](https://github.com/generative-computing/mellea)

#### 1. The `@generative` Pattern
**Don't** write prompt templates or regex parsers:
```python
# BAD - don't do this
response = openai.chat.completions.create(...)
age = int(re.search(r"\d+", response).group())
```
**Do** use typed function signatures:
```python
from mellea import generative, start_session

@generative
def extract_age(text: str) -> int:
    """Extract the user's age from text."""
    ...

m = start_session()
age = extract_age(m, text="Alice is 30")  # Returns int(30)
```

#### 2. Complex Types
```python
from pydantic import BaseModel
from mellea import generative

class UserProfile(BaseModel):
    name: str
    age: int
    interests: list[str]

@generative
def parse_profile(bio: str) -> UserProfile: ...
```

#### 3. Chain-of-Thought
Add `reasoning` field to force the LLM to "think" before answering:
```python
from typing import Literal
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    reasoning: str  # LLM fills first
    conclusion: Literal["approve", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)

@generative
def analyze_document(doc: str) -> AnalysisResult: ...
```

#### 4. Control Flow
Use Python `if/for/while`. No graph frameworks needed:
```python
if analyze_sentiment(m, email) == "negative":
    draft = draft_apology(m, email)
else:
    draft = draft_response(m, email)
```

#### 5. Instruct-Validate-Repair
For strict requirements, use `m.instruct()`:
```python
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

email = m.instruct(
    "Write an invite for {{name}}",
    requirements=[
        req("Must be formal"),
        req("Lowercase only", validation_fn=simple_validate(lambda x: x.islower()))
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    user_variables={"name": "Alice"}
)
```

#### 6. Small Model Fix
Small models (1B-8B) can't calculate. Extract params with LLM, compute in Python:
```python
from pydantic import BaseModel

class PhysicsParams(BaseModel):
    speed_a: float
    speed_b: float
    delay_hours: float

@generative
def extract_params(text: str) -> PhysicsParams:
    """EXTRACT numbers only. Do not calculate."""
    ...

def calculate_gap(p: PhysicsParams) -> float:
    return p.speed_a * p.delay_hours
```

#### 7. One-Shot Examples
If model struggles, add examples to docstring:
```python
@generative
def identify_fruit(text: str) -> str | None:
    """
    Extract fruit from text, or None if none mentioned.
    Ex: "I ate an apple" -> "apple"
    Ex: "The sky is blue" -> None
    """
    ...
```

#### 8. Backend Config
```python
from mellea import start_session
from mellea.backends.model_options import ModelOption

m = start_session(
    model_id="granite3.3:8b",
    model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 500}
)
```
Options: `TEMPERATURE`, `MAX_NEW_TOKENS`, `SYSTEM_PROMPT`, `SEED`, `TOOLS`, `CONTEXT_WINDOW`, `THINKING`, `STREAM`

#### 9. Async
```python
@generative
async def extract_age(text: str) -> int:
    """Extract age."""
    ...

result = await extract_age(m, text="Alice is 30")
```
Session methods: `ainstruct`, `achat`, `aact`, `avalidate`, `aquery`, `atransform`

#### 10. Auth
- **Ollama**: `start_session()` (no setup)
- **OpenAI**: `export OPENAI_API_KEY="..."`
- **Watsonx**: `export WATSONX_API_KEY="..."`, `WATSONX_URL`, `WATSONX_PROJECT_ID`
- **Bedrock**: `export AWS_BEARER_TOKEN_BEDROCK="..."`

**Never hardcode API keys.**

#### 11. Tool Calling
Define tools with `@tool` and pass them via `ModelOption.TOOLS`:
```python
from mellea.backends import tool
from mellea.backends.model_options import ModelOption

@tool
def get_weather(location: str, days: int = 1) -> dict:
    """Get weather forecast for a location.

    Args:
        location: City name
        days: Number of days to forecast
    """
    return {"location": location, "forecast": "sunny"}

response = m.instruct(
    description="What's the weather in Boston?",
    model_options={ModelOption.TOOLS: [get_weather]},
)
```

#### 12. Multi-Turn Chat
Use `ChatContext` for conversation history:
```python
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
m.chat("Make up a math problem.")
m.chat("Now solve it.")
print(m.ctx.last_output())
```
`ChatContext(window_size=10)` limits history to the last N turns.

#### 13. ReACT Agent
Agentic reasoning loop with tool use:
```python
import asyncio
from mellea import start_session
from mellea.backends import tool
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.context import ChatContext

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"

async def main():
    m = start_session()
    output, context = await react(
        goal="What is the Mellea library?",
        context=ChatContext(),
        backend=m.backend,
        tools=[search],
    )
    print(output)

asyncio.run(main())
```

#### 14. Plugins
Hook into the generation lifecycle for logging, policy enforcement, or observability:
```python
from mellea.plugins import hook, register, HookType

@hook(HookType.GENERATION_PRE_CALL)
async def log_generation(payload, ctx):
    print(f"Calling LLM with: {payload.action}")

register(log_generation)
```
Hook types: `GENERATION_PRE_CALL`, `GENERATION_POST_CALL`, `SESSION_PRE_INIT`, `SESSION_POST_INIT`

#### 15. Anti-Patterns
- **Don't** retry `@generative` calls — Mellea handles retries internally
- **Don't** use `json.loads()` — use typed returns
- **Don't** wrap single functions in classes
- **Do** use `try/except` at app boundaries for network errors

#### 16. Debugging
```python
from mellea.core import FancyLogger
FancyLogger.get_logger().setLevel("DEBUG")
```
- `m.last_prompt()` — see exact prompt sent

#### 17. Common Errors
| Error | Fix |
|-------|-----|
| `ComponentParseError` | LLM output didn't match type—add docstring examples |
| `TypeError: missing positional argument` | First arg must be session `m` |
| `ConnectionRefusedError` | Run `ollama serve` |
| Output wrong/None | Model too small—try larger or add `reasoning` field |

#### 18. Testing
```bash
uv run pytest test/ -m "not qualitative"  # Fast: tests only, skip quality checks
uv run pytest                              # Full: tests + examples + quality checks
```

#### 19. Feedback
Found a workaround or pattern? Add it to Section 17 (Common Errors) above, or update this file with new guidance.
