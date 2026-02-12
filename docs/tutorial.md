# Principles of Generative Programming: The Mellea Approach

## Table of Contents

- [Principles of Generative Programming: The Mellea Approach](#principles-of-generative-programming-the-mellea-approach)
  - [Table of Contents](#table-of-contents)
  - [Chapter 1: What Is Generative Programming](#chapter-1-what-is-generative-programming)
  - [Chapter 2: Getting Started with Generative Programming in Mellea](#chapter-2-getting-started-with-generative-programming-in-mellea)
    - [Requirements](#requirements)
    - [Validating Requirements](#validating-requirements)
    - [Instruct - Validate - Repair](#instruct---validate---repair)
    - [ModelOptions](#modeloptions)
      - [System Messages](#system-messages)
    - [Conclusion](#conclusion)
  - [Chapter 3: Overview of the Standard Library](#chapter-3-overview-of-the-standard-library)
  - [Chapter 4: Generative Slots](#chapter-4-generative-slots)
      - [Example: Sentiment Classifier](#example-sentiment-classifier)
      - [Using Generative slots to Provide Compositionality Across Module Boundaries](#using-generative-slots-to-provide-compositionality-across-module-boundaries)
  - [Chapter 5: MObjects](#chapter-5-mobjects)
    - [Example: A Table as an MObject](#example-a-table-as-an-mobject)
    - [Case Study: Working with Documents](#case-study-working-with-documents)
    - [MObject methods are tools](#mobject-methods-are-tools)
  - [Chapter 6: Tuning Requirements and Components](#chapter-6-tuning-requirements-and-components)
    - [Problem Statement](#problem-statement)
    - [Training the aLoRA Adapter](#training-the-alora-adapter)
      - [Parameters](#parameters)
    - [Upload to Hugging Face (Optional)](#upload-to-hugging-face-optional)
    - [Generating a README for Your Adapter](#generating-a-readme-for-your-adapter)
    - [Integrating the Tuned Model into Mellea](#integrating-the-tuned-model-into-mellea)
  - [Chapter 7: On Context Management](#chapter-7-on-context-management)
  - [Chapter 8: Implementing Agents](#chapter-8-implementing-agents)
    - [Case Study: Implementing ReACT in Mellea](#case-study-implementing-react-in-mellea)
    - [Guarded Nondeterminism](#guarded-nondeterminism)
  - [Chapter 9: Interoperability with Other Frameworks](#chapter-9-interoperability-with-other-frameworks)
    - [Simple mcp server running Mellea](#simple-mcp-server-running-mellea)
    - [Running Mellea programs as an openai compatible server (Experimental)](#running-mellea-programs-as-an-openai-compatible-server-experimental)
      - [Example `m serve` application](#example-m-serve-application)
  - [Chapter 10: Prompt Engineering for M](#chapter-10-prompt-engineering-for-m)
    - [Templates](#templates)
    - [Template Representations](#template-representations)
    - [Customization](#customization)
      - [Choosing a Template](#choosing-a-template)
      - [Editing an Existing Class](#editing-an-existing-class)
    - [Large Prompts and Decompose](#decompose)
  - [Chapter 11: Tool Calling](#chapter-11-tool-calling)
  - [Chapter 12: Asynchronicity](#chapter-12-asynchronicity)
    - [Asynchronous Functions:](#asynchronous-functions)
    - [Asynchronicity in Synchronous Functions](#asynchronicity-in-synchronous-functions)
  - [Appendix: Contributing to Mellea](#appendix-contributing-to-mellea)
    - [Contributor Guide: Getting Started](#contributor-guide-getting-started)
    - [Contributor Guide: Requirements and Verifiers](#contributor-guide-requirements-and-verifiers)
    - [Contributor Guide: Components](#contributor-guide-components)
    - [Contributor Guide: Specialized Mify](#contributor-guide-specialized-mify)
    - [Contributor Guide: Sessions](#contributor-guide-sessions)

## Chapter 1: What Is Generative Programming

This tutorial is about Mellea. Mellea helps you write better generative programs.

A *generative program* is any computer program that contains calls to an LLM. As we will see throughout the tutorial, LLMs can be incorporated into software in a wide variety of ways. Some ways of incorporating LLMs into programs tend to result in robust and performant systems, while others result in software that is brittle and error-prone.

Generative programs are distinguished from classical programs by their use of functions that invoke generative models. These generative calls can produce many different data types -- strings, booleans, structured data, code, images/video, and so on. The model(s) and software underlying generative calls can be combined and composed in certain situations and in certain ways (e.g., LoRA adapters as a special case). In addition to invoking generative calls, generative programs can invoke other functions, written in languages that do not have an LLM in their base, so that we can, for example, pass the output of a generative function into a DB retrieval system and feed the output of that into another generator. Writing generative programs is difficult because generative programs interleave deterministic and stochastic operations.

Requirement verification plays an important role in circumscribing periods of nondeterminism in a generative program.  We can implement validators that produce boolean or other outputs, and repeat loops until the validator says yes, or perhaps the iteration count gets too high and we trigger some exception handling process. Thus we can determine the degree of certainty in the output of a generative function and then act based upon the amount of certainty. Verification can happen in a variety of ways -- from querying a generative function, to precise programmatic checks, and a variety of combinations besides.

In programs that contain long computation paths -- including most that contain iteration or recursion -- incremental accrual of uncertainty is multiplicative, and therefore must itself be occasionally circumscribed by incremental requirement verification throughout the generative program's execution. These incremental checks can be used to establish patterns of variation, or properties which are invariant, both of which can help ensure that the execution converges to a desired state and does not "go wrong". The construction of these incremental checks is one of the important tasks in generative programming, and can itself be treated as a task amenable to generative programming. Like other requirement checks, these variants and invariants may be explicit and programmatic or can be solved via a generative function. In any case, each generative program results in a trace of computations -- some successful, others failures.

Figuring out what to do about failure paths is yet another crux faced by authors of generative programs. Successful traces can be collected, leading to a final high-confidence result; alternatively, traces with some failures or low-confidence answers can accumulate. Generative programs then try to repair these failed validations. The repair process can be manual, or automated, or offer a combination of user interactions and automated repair mechanisms. As a generative program executes in this way, context accrues. The accrual of ever-larger contexts becomes a challenge unto itself.

Memory management therefore plays an important role in context engineering. Mellea therefore provides a mechanism for mapping components of KV Cache onto developer and user-facing abstractions, and for automating the construction of context and handling of cached keys and values.

As the Mellea developers built this library for generative programming, we found some useful principles that you will see re-occur throughout this tutorial:

 * **circumscribe LLM calls with requirement verifiers.** We will see variations on this principle throughout the tutorial.
 * **Generative programs should use simple and composable prompting styles.** Mellea takes a middle-ground between the "framework chooses the prompt" and "client code chooses the prompt" paradigms. By keeping prompts small and self-contained, then chaining together many such prompts, we can usually get away with one of a few prompt styles. When a new prompt style is needed, that prompt should be co-designed with the software that will use the prompt. In Mellea, we encourage this by decomposing generative programs into *Components*; more on this in [Chapter 3](#chapter-3-overview-of-the-standard-library).
 * **Generative models and inference-time programs should be co-designed.** Ideally, the style and domain of prompting used at inference time should match the style and domain of prompting using in pretraining, mid-training, and/or post-training. And, similarly, models should be built with runtime components and use-patterns in mind. We will see some early examples of this in [Chapter 6](#chapter-6-tuning-requirements-and-components).
 * **Generative programs should carefully manage context.** Each Component manages context of a single call, as we see in Chapters [2](#chapter-2-getting-started-with-generative-programming-in-mellea), [3](#chapter-3-overview-of-the-standard-library), [4](#chapter-4-generative-slots), and [5](#chapter-5-mobjects). Additionally, Mellea provides some useful mechanisms for re-using context across multiple calls ([Chapter 7](#chapter-7-on-context-management)).

Although good generative programs can be written in any language and framework, getting it right is not trivial. Mellea is just one point in the design space of LLM libraries, but we think it is a good one. Our hope is that Mellea will help you write generative programs that are robust, performant, and fit-for-purpose.

## Chapter 2: Getting Started with Generative Programming in Mellea

Before we get started, you will need to download and install [ollama](https://ollama.com/). Mellea can work with many different types of backends, but everything in this tutorial will "just work" on a Macbook running IBM's Granite 4 Micro 3B model.

We also recommend that you download and install [uv](https://docs.astral.sh/uv/#installation). You can run any of the examples in the tutorial with:
```bash
uv run example_name.py --with mellea
```
> [!NOTE]
> If running on an Intel mac, you may get errors related to torch/torchvision versions. Conda maintains updated versions of these packages. You will need to create a conda environment and run `conda install 'torchvision>=0.22.0'` (this should also install pytorch and torchvision-extra). Then, you should be able to run `uv pip install mellea`. To run the examples, you will need to use `python <filename>` inside the conda environment instead of `uv run --with mellea <filename>`.

> [!NOTE]
> If you are using python >= 3.13, you may encounter an issue where outlines cannot be installed due to rust compiler issues (`error: can't find Rust compiler`). You can either downgrade to python 3.12 or install the [rust compiler](https://www.rust-lang.org/tools/install) to build the wheel for outlines locally.

Once you have ollama installed and running, we can get started with our first generative piece of code:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L1-L8
import mellea

# INFO: this line will download IBM's Granite 4 Micro 3B model.
m = mellea.start_session()

email = m.instruct("Write an email inviting interns to an office party at 3:30pm.")
print(str(email))
```

Here, we initialized a backend running Ollama on a local machine using the granite3.3-chat model.
We then ask the model to generate an email and print it to the console.

> [!NOTE]
> Mellea supports many other models and backends. By default, a new Mellea session will run IBM's capable Granite 3B model on your own laptop. This is a good (and free!) way to get started. If you would like to try out other models or backends, you can explicitly specify the backend and model in the start_session method. For example, `mellea.start_session(backend_name="ollama", model_id=mellea.model_ids.IBM_GRANITE_4_MICRO_3B)`.

Before continuing, let's wrap this call into a function with some arguments:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L13-L27
import mellea

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
  email = m.instruct(
    "Write an email to {{name}} using the notes following: {{notes}}.",
    user_variables={"name": name, "notes": notes},
  )
  return email.value  # str(email) also works.

m = mellea.start_session()
print(write_email(m, "Olivia",
                  "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery."))
```

Voila, we now have an email-writing function!

Notice how the instruct method can take a dictionary of variables as `user_variables`. These are filled by treating the instruction description as a jinja template.

The `m.instruct()` function returns a `ModelOutputThunk` per default, which has the model output string bound to the field `.value`.

### Requirements

But how do we know that the generated email is a good one?
Good generative programmers don't leave this up to chance -- instead, they use pre-conditions to ensure that inputs to the LLM are as expected and then check post-conditions to ensure that the LLM's outputs are fit-for-purpose.

Suppose that in this case we want to ensure that the email has a salutation and contains only lower-case letters. We can capture these post-conditions by specifying **requirements** on the `m.instruct` call:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L33-L53
import mellea

def write_email_with_requirements(m: mellea.MelleaSession, name: str, notes: str) -> str:
  email = m.instruct(
      "Write an email to {{name}} using the notes following: {{notes}}.",
      requirements=[
          "The email should have a salutation",
          "Use only lower-case letters",
      ],
      user_variables={"name": name, "notes": notes},
  )
  return str(email)

m = mellea.start_session()
print(write_email_with_requirements(
  m,
  name="Olivia",
  notes="Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
))
```

We just added two requirements to the instruction which will be added to the model request. But we don't check yet if these requirements are satisfied. Let's add a **strategy** for validating the requirements:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py#L57-L84
import mellea
from mellea.stdlib.sampling import RejectionSamplingStrategy

def write_email_with_strategy(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation",
            "Use only lower-case letters",
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )
    if email_candidate.success:
        return str(email_candidate.result)
    else:
        print("Expect sub-par result.")
        return email_candidate.sample_generations[0].value

m = mellea.start_session()
print(
    write_email_with_strategy(
        m,
        "Olivia",
        "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery.",
    )
)
```

A couple of things happened here. First, we added a sampling `strategy` to the instruction.
This strategy (`RejectionSamplingStrategy()`) checks if all requirements are met.
If any requirement fails, then the sampling strategy will sample a new email from the LLM.
This process will repeat until the `loop_budget` on retries is consumed or all requirements are met.

Even with retries, sampling might not generate results that fulfill all requirements (`email_candidate.success==False`).
Mellea forces you to think about what it means for an LLM call to fail;
in this case, we handle the situation by simply returning the first sample as the final result.

> [!NOTE]
> When using the `return_sampling_results=True` parameter, the `instruct()` function returns a `SamplingResult` object (not a `ModelOutputThunk`) which carries the full history of sampling and validation results for each sample.

### Validating Requirements

Now that we defined requirements and sampling we should have a
look into **how requirements are validated**. The default validation strategy is [LLM-as-a-judge](https://arxiv.org/abs/2306.05685).

Let's look on how we can customize requirement definitions:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/instruct_validate_repair.py#L1-L10
from mellea.stdlib.requirements import req, check, simple_validate

requirements = [
    req("The email should have a salutation"),  # == r1
    req("Use only lower-case letters", validation_fn=simple_validate(lambda x: x.lower() == x)),  # == r2
    check("Do not mention purple elephants.")  # == r3
]
```
Here, the first requirement (r1) will be validated by LLM-as-a-judge on the output (last turn) of the instruction. This is the default behavior, since nothing else is specified.

The second requirement (r2) simply  uses a function that takes the output of a sampling step and returns a boolean value indicating (un-)successful validation. While the `validation_fn` parameter requires to run validation on the full session context (see [Chapter 7](#chapter-7-on-context-management)), Mellea provides a wrapper for simpler validation functions (`simple_validate(fn: Callable[[str], bool])`) that take the output string and return a boolean as seen in this case.

The third requirement is a `check()`. Checks are only used for validation, not for generation.
Checks aim to avoid the "do not think about B" effect that often primes models (and humans)
to do the opposite and "think" about B.

> [!NOTE]
> LLMaJ is not presumptively robust. Whenever possible, implement requirement validation using plain old Python code. When a model is necessary, it can often be a good idea to train a **calibrated** model specifically for your validation problem. [Chapter 6](#chapter-6-tuning-requirements-and-components) explains how to use Mellea's `m tune` subcommand to train your own LoRAs for requirement checking (and for other types of Mellea components as well).


### Instruct - Validate - Repair

Now, we bring it all together into a first generative program using the **instruct-validate-repair** pattern:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/instruct_validate_repair.py#L13-L37
import mellea
from mellea.stdlib.requirements import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            req("The email should have a salutation"),  # == r1
            req(
                "Use only lower-case letters",
                validation_fn=simple_validate(lambda x: x.lower() == x),
            ),  # == r2
            check("Do not mention purple elephants."),  # == r3
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"name": name, "notes": notes},
        return_sampling_results=True,
    )
    if email_candidate.success:
        return str(email_candidate.result)
    else:
        return email_candidate.sample_generations[0].value


m = mellea.start_session()
print(write_email(m, "Olivia",
                  "Olivia helped the lab over the last few weeks by organizing intern events, advertising the speaker series, and handling issues with snack delivery."))
```

> [!NOTE]
> The `instruct()` method is a convenience function that creates and then generates from an `Instruction` Component, `req()` similarly wraps the `Requirement` Component, etc. [Chapter 2](#chapter-2-getting-started-with-generative-programming-in-mellea) will takes us one level deeper into understanding what happens under the hood when you call `m.instruct()`.


### ModelOptions

Most LLM apis allow you to specify options to modify the request: temperature, max_tokens, seed, etc... Mellea supports specifying these options during backend initialization and when calling session-level functions with the `model_options` parameter.

Mellea supports many different types of inference engines (ollama, openai-compatible vllm, huggingface, etc.). These inference engines, which we call `Backend`s, provide different and sometimes inconsistent dict keysets for specifying model options. For the most common options among model providers, Mellea provides some engine-agnostic options, which can be used by typing [`ModelOption.<TAB>`](../mellea/backends/types.py) in your favorite IDE; for example, temperature can be specified as `{"{ModelOption.TEMPERATURE": 0}` and this will "just work" across all inference engines.

You can add any key-value pair supported by the backend to the `model_options` dictionary, and those options will be passed along to the inference engine *even if a Mellea-specific `ModelOption.<KEY>` is defined for that option. This means you can safely copy over model option parameters from exiting codebases as-is:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/model_options_example.py#L1-L16
import mellea
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids

m = mellea.MelleaSession(backend=OllamaModelBackend(
    model_id=model_ids.IBM_GRANITE_3_2_8B,
    model_options={ModelOption.SEED: 42}
))

answer = m.instruct(
    "What is 2x2?",
    model_options={
        "temperature": 0.5,
        "num_predict": 5,
    },
)

print(str(answer))
```

You can always update the model options of a given backend; however, Mellea offers a few additional approaches to changing the specified options.

1. **Specifying options during `m.*` calls**. Options specified here will update the model options previously specified for that call only. If you specify an already existing key (with either the `ModelOption.OPTION` version or the native name for that option for the given api), the value will be the one associated with the new key. If you specify the same key in different ways (ie `ModelOption.TEMPERATURE` and `temperature`), the `ModelOption.OPTION` key will take precedence.

```python
# options passed during backend initialization
backend_model_options = {
    "seed": "1",
    ModelOption.MAX_NEW_TOKENS: 1,
    "temperature": 1,
}

# options passed during m.*
instruct_model_options = {
    "seed": "2",
    ModelOption.SEED: "3",
    "num_predict": 2,
}

# options passed to the model provider API
final_options = {
    "temperature": 1,
    "seed": 3,
    "num_predict": 2
}
```

2. **Pushing and popping model state**. Sessions offer the ability to push and pop model state. This means you can temporarily change the `model_options` for a series of calls by pushing a new set of `model_options` and then revert those changes with a pop.

#### System Messages
In Mellea, `ModelOption.SYSTEM_PROMPT` is the recommended way to add/change the system message for a prompt. Setting it at the backend/session level will use the provided message as the system prompt for all future calls (just like any other model option). Similarly, you can specify the system prompt parameter for any session-level function (like `m.instruct`) to replace it for just that call.

Mellea recommends applying the system message this way because some model-provider apis don't properly serialize messages with the `system` role and expect them as a separate parameter.

### Conclusion

We have now worked up from a simple "Hello, World" example to our first generative programming design pattern: **Instruct - Validate - Repair (IVR)**.

When LLMs work well, the software developer experiences the LLM as a sort of oracle that can handle most any input and produce a sufficiently desirable output. When LLMs do not work at all, the software developer experiences the LLM as a naive markov chain that produces junk. In both cases, the LLM is just sampling from a distribution.

The crux of generative programming is that most applications find themselves somewhere in-between these two extremes -- the LLM mostly works, enough to demo a tantalizing MVP. But failure modes are common enough and severe enough that complete automation is beyond the developer's grasp.

Traditional software deals with failure modes by carefully describing what can go wrong and then providing precise error handling logic. When working with LLMs, however, this approach suffers a Sysiphean curse. There is always one more failure mode, one more special case, one more new feature request. In the next chapter, we will explore how to build generative programs that are compositional and that grow gracefully.

## Chapter 3: Overview of the Standard Library

Before going any further, we need to overview the architecture of Mellea.

Mellea's core abstraction is called a `Component`. A `Component` is a structured object that represents a unit of interaction with an LLM. The Mellea `stdlib` contains a set of useful components, but you can also define your own.  We have already seen some components -- `Instruction` and `Requirement` are both `Component`s.

Components are composite data structures; that is, a `Component` can be made up of many other parts. Each of those parts is either a `CBlock` or another `Component`. `CBlock`s, or "content blocks", are an atomic unit of text or data. CBlocks hold raw text (or sometimes parsed representations) and can be used as leaves in the Component DAG.

Components can also specify an expected output type along with a parse function to extract that type from the LLM's output. By default, this type is a string; but by defining a Component's expected type, you can get type hinting for outputs in the standard library.

Backends are the engine that actually run the LLM. Backends consume Components, format the Component, pass the formatted input to an LLM, and return model outputs, which are then parsed back into CBlocks or Components.

During the course of an interaction with an LLM, several Components and CBlocks may be created. Logic for handling this trace of interactions is provided by a `Context` object. Some book-keeping needs to be done in order for Contexts to appropriately handle a trace of Components and CBlocks. The `MelleaSession` class, which is created by `mellea.start_session()`, does this book-keeping a simple wrapper around Contexts and Backends.

When we call `m.instruct()`, the `MelleaSession.instruct` method creates a component called an `Instruction`. Instructions are part of the Mellea standard library.

So far we have seen Instructions with descriptions and requirements, but an Instruction can also have in-context learning examples and grounding_context (for RAG):

```python
class Instruction(Component):
    """The Instruction in an instruct/validate/repair loop."""

    def __init__(
        self,
        description: str | CBlock | None = None,
        requirements: list[Requirement | str] | None = None,
        icl_examples: list[str | CBlock] | None = None,
        grounding_context: dict[str, str | CBlock | Component] | None = None,
        user_variables: dict[str, str] | None = None,
        prefix: str | CBlock | None = None,
        output_prefix: str | CBlock | None = None,
    ):
```

The following Cheat Sheet concisely visualizes the relationship between Components/CBlocks, Backends, Contexts, and Sessions.

TODO INSERT HENDRIK'S CHEAT SHEET

M's standard library contains four basic types of Components:

1. [Instructions](#chapter-2-getting-started-with-generative-programming-in-mellea), which we have already seen.
2. [Requirements](#chapter-2-getting-started-with-generative-programming-in-mellea), which we have already seen and will continue to use heavily throughout the remainder of the tutorial.
3. [Generative Slots](#chapter-4-generative-slots), which treat LLM calls as functions.
4. [MObjects](#chapter-5-mobjects), which help with context engineering for tool use by placing tools next to the data that those tools most reasonably operate over.

This is not an exhaustive list of possible component types. New components can be created as [user libraries or as stdlib contributions](#appendix-contributing-to-m). Where it makes sense, you can also back new components by [fine-tuned models designed especially to work with your Component types](#chapter-6-tuning-requirements-and-components).

But before getting into these advanced modalities, let's finish our overview of the standard library of Components that ship with Mellea.

## Chapter 4: Generative Slots

In classical programming, pure (stateless) functions are a simple and powerful abstraction. A pure function takes inputs, computes outputs, and has no side effects. Generative programs can also use functions as abstraction boundaries, but in a generative program the meaning of the function can be given by an LLM instead of an interpreter or compiler. This is the idea behind a **GenerativeSlot**.

A `GenerativeSlot` is a function whose implementation is provided by an LLM. In Mellea, you define these using the `@generative` decorator. The function signature specifies the interface, and the docstring (or type annotations) guide the LLM in producing the output.

#### Example: Sentiment Classifier

Let's start with a simple example: a function that classifies the sentiment of a string as "positive" or "negative".

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/sentiment_classifier.py#L1-L13
from typing import Literal
from mellea import generative, start_session

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
  """Classify the sentiment of the input text as 'positive' or 'negative'."""
  ...

m = start_session()
sentiment = classify_sentiment(m, text="I love this!")
print("Output sentiment is:", sentiment)
```

Here, `classify_sentiment` is a GenerativeSlot: it looks like a normal function, but its implementation is handled by the LLM. The type annotation (`Literal["positive", "negative"]`) constrains the output, and the prompt is automatically constructed from the function signature and docstring.

Many more examples of generative slots are provided in the `docs/examples` directory.

> [!NOTE]
> Generative slots can also be implemented as code-generation calls instead of black-box structured output generators. This is most useful when correct code generation is difficult without some dynamic analysis (i.e., runtime information). In these cases, the problem can be solved by prompting with a FiTM code generation request, augmented with pieces of runtime state. This advanced functionality may result in untrusted code execution, and should therefore be used with caution and/or in conjunction with some combination of sandboxing and human validation prior to execution.

#### Using Generative slots to Provide Compositionality Across Module Boundaries

Instruct-validate-repair provides compositionality within a given module. As the examples listed above demonstrate, generative slots can do the same. But generative slots are not just about local validity; their real power comes from safe interoperability between independently designed systems.

Consider the following two independently developed libraries: a **Summarizer** library that contains a set of functions for summarizing various types of documents, and a **Decision Aids** library that aids in decision making for particular situations.

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L1-L18
from mellea import generative

# The Summarizer Library
@generative
def summarize_meeting(transcript: str) -> str:
  """Summarize the meeting transcript into a concise paragraph of main points."""

@generative
def summarize_contract(contract_text: str) -> str:
  """Produce a natural language summary of contract obligations and risks."""

@generative
def summarize_short_story(story: str) -> str:
  """Summarize a short story, with one paragraph on plot and one paragraph on broad themes."""
```

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L20-L33
from mellea import generative

# The Decision Aides Library
@generative
def propose_business_decision(summary: str) -> str:
  """Given a structured summary with clear recommendations, propose a business decision."""

@generative
def generate_risk_mitigation(summary: str) -> str:
  """If the summary contains risk elements, propose mitigation strategies."""

@generative
def generate_novel_recommendations(summary: str) -> str:
  """Provide a list of novel recommendations that are similar in plot or theme to the short story summary."""
```

Notice that these two libraries do not necessarily always compose -- meeting notes may or may not contain semantic content for which risk analysis even makes sense.

To help us compose these libraries, we introduce a set of contracts that gate function composition and then use those contracts to short-circuit non-sensical compositions of library components:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L36-L52
from mellea import generative
from typing import Literal

# Compose the libraries.
@generative
def has_structured_conclusion(summary: str) -> Literal["yes", "no"]:
  """Determine whether the summary contains a clearly marked conclusion or recommendation."""

@generative
def contains_actionable_risks(summary: str) -> Literal["yes", "no"]:
  """Check whether the summary contains references to business risks or exposure."""

@generative
def has_theme_and_plot(summary: str) -> Literal["yes", "no"]:
  """Check whether the summary contains both a plot and thematic elements."""
```

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/compositionality_with_generative_slots.py#L55-L129
from mellea import start_session

m = start_session()
transcript = """Meeting Transcript: Market Risk Review -- Self-Sealing Stembolts Division
Date: December 1, 3125
Attendees:

Karen Rojas, VP of Product Strategy

Derek Madsen, Director of Global Procurement

Felicia Zheng, Head of Market Research

Tom Vega, CFO

Luis Tran, Engineering Liaison

Karen Rojas:
Thanks, everyone, for making time on short notice. As you've all seen, we've got three converging market risks we need to address: tariffs on micro-carburetors, increased adoption of the self-interlocking leafscrew, and, believe it or not, the "hipsterfication" of the construction industry. I need all on deck and let's not waste time. Derek, start.

Derek Madsen:
Right. As of Monday, the 25% tariff on micro-carburetors sourced from the Pan-Alpha Centauri confederacy is active. We tried to pre-purchase a three-month buffer, but after that, our unit cost rises by $1.72. That's a 9% increase in the BOM cost of our core model 440 stembolt. Unless we find alternative suppliers or pass on the cost, we're eating into our already narrow margin.

Tom Vega:
We cannot absorb that without consequences. If we pass the cost downstream, we risk losing key mid-tier OEM clients. And with the market already sniffing around leafscrew alternatives, this makes us more vulnerable.

Karen:
Lets pause there. Felicia, give us the quick-and-dirty on the leafscrew.

Felicia Zheng:
It's ugly. Sales of the self-interlocking leafscrew—particularly in modular and prefab construction—are up 38% year-over-year. It's not quite a full substitute for our self-sealing stembolts, but they are close enough in function that some contractors are making the switch. Their appeal? No micro-carburetors, lower unit complexity, and easier training for install crews. We estimate we've lost about 12% of our industrial segment to the switch in the last two quarters.

Karen:
Engineering, Luis; your take on how real that risk is?

Luis Tran:
Technically, leafscrews are not as robust under high-vibration loads. But here's the thing: most of the modular prefab sites don not need that level of tolerance. If the design spec calls for durability over 10 years, we win. But for projects looking to move fast and hit 5-year lifespans? The leafscrew wins on simplicity and cost.

Tom:
So they're eating into our low-end. That's our volume base.

Karen:
Exactly. Now let's talk about this last one: the “hipsterfication” of construction. Felicia?

Felicia:
So this is wild. We're seeing a cultural shift in boutique and residential construction—especially in markets like Beckley, West Sullivan, parts of Osborne County, where clients are requesting "authentic" manual fasteners. They want hand-sealed bolts, visible threads, even mismatched patinas. It's an aesthetic thing. Function is almost secondary. Our old manual-seal line from the 3180s? People are hunting them down on auction sites.

Tom:
Well, I'm glad I don't have to live in the big cities... nothing like this would ever happen in downt-to-earth places Brooklyn, Portland, or Austin.

Luis:
We literally got a request from a design-build firm in Keough asking if we had any bolts “pre-distressed.”

Karen:
Can we spin this?

Tom:
If we keep our vintage tooling and market it right, maybe. But that's niche. It won't offset losses in industrial and prefab.

Karen:
Not yet. But we may need to reframe it as a prestige line—low volume, high margin. Okay, action items. Derek, map alternative micro-carburetor sources. Felicia, get me a forecast on leafscrew erosion by sector. Luis, feasibility of reviving manual seal production. Tom, let's scenario-plan cost pass-through vs. feature-based differentiation.

Let's reconvene next week with hard numbers. Thanks, all."""
summary = summarize_meeting(m, transcript=transcript)

if contains_actionable_risks(m, summary=summary) == "yes":
    mitigation = generate_risk_mitigation(m, summary=summary)
    print(f"Mitigation: {mitigation}")
else:
    print("Summary does not contain actionable risks.")
if has_structured_conclusion(m, summary=summary) == "yes":
    decision = propose_business_decision(m, summary=summary)
    print(f"Decision: {decision}")
else:
    print("Summary lacks a structured conclusion.")
```

Without these Hoare-style contracts, the only way to ensure composition is to couple the libraries, either by rewriting `summarize_meeting` to conform to `propose_business_decision`, or adding Requirements to `propose_business_decision` that may silently fail if unmet. These approaches can work, but require tight coupling between these two otherwise loosely coupled libraries.

With contracts, we **decouple** the libraries without sacrificing safe dynamic composition, by moving the coupling logic into pre- and post-condition checks. This is another LLM-native software engineering pattern: **guarded nondeterminism**.

## Chapter 5: MObjects

Object-oriented programming (OOP) is a powerful paradigm for organizing code: you group related data and the methods that operate on that data into classes. In the world of LLMs, a similar organizational principle emerges—especially when you want to combine structured data with LLM-powered "tools" or operations. This is where Mellea's **MObject** abstraction comes in.

**The MObject Pattern:** You should store data alongside its relevant operations (tools). This allows LLMs to interact with both the data and methods in a unified, structured manner. It also simplifies the process of exposing only the specific fields and methods you want the LLM to access.

The `MOBject` pattern also provides a way of evolving existing classical codebases into generative programs. Mellea's `@mify` decorator lets you turn **any** class into an `MObject`. If needed, you can specify which fields and methods are included, and provide a template for how the object should be represented to the LLM.

### Example: A Table as an MObject

Suppose you have a table of sales data and want to let the LLM answer questions about it:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/table_mobject.py#L1-L31
import mellea
from mellea.stdlib.components.mify import mify, MifiedProtocol
import pandas
from io import StringIO


@mify(fields_include={"table"}, template="{{ table }}")
class MyCompanyDatabase:
  table: str = """| Store      | Sales   |
                    | ---------- | ------- |
                    | Northeast  | $250    |
                    | Southeast  | $80     |
                    | Midwest    | $420    |"""

  def transpose(self):
    pandas.read_csv(
      StringIO(self.table),
      sep='|',
      skipinitialspace=True,
      header=0,
      index_col=False
    )


m = mellea.start_session()
db = MyCompanyDatabase()
assert isinstance(db, MifiedProtocol)
answer = m.query(db, "What were sales for the Northeast branch this month?")
print(str(answer))
```

In this example, the `@mify` decorator transforms MyCompanyDatabase into an MObject. Only the *table* field is incorporated into the Large Language Model (LLM) prompt, as designated by `fields_include`. The `template` describes how the object is presented to the model. The `.query()` method now enables you to pose questions about the data, allowing the LLM to utilize the table as contextual information.


**When to use MObjects?**
MObjects offer a sophisticated and modular approach to linking structured data with operations powered by Large Language Models (LLMs). They provide precise control over what the LLM can access, allowing for the exposure of custom tools or methods. This design pattern can be particularly useful for tool-calling, document querying, and any scenario where data needs to be "wrapped" with behaviors accessible to an LLM.

We'll see more advanced uses of MObjects -- including tool registration and custom operations -- in our next case study on working with rich-text documents.

### Case Study: Working with Documents

Mellea makes it easy to work with documents. For that we provide `mified` wrappers
around [docling](https://github.com/docling-project/docling) documents.

Let's create a RichDocument from an arxiv paper:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py#L1-L3
from mellea.stdlib.components.docs.richdocument import RichDocument
rd = RichDocument.from_document_file("https://arxiv.org/pdf/1906.04043")
```
this loads the PDF file and parses it using the Docling parser into an
intermediate representation.

From the rich document we can extract some document content, e.g. the
first table:
```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py#L5-L8
from mellea.stdlib.components.docs.richdocument import Table
table1: Table = rd.get_tables()[0]
print(table1.to_markdown())
```
Output:
```markdown
| Feature                              | AUC         |
|--------------------------------------|-------------|
| Bag of Words                         | 0.63 ± 0.11 |
| (Test 1 - GPT-2) Average Probability | 0.71 ± 0.25 |
| (Test 2 - GPT-2) Top-K Buckets       | 0.87 ± 0.07 |
| (Test 1 - BERT) Average Probability  | 0.70 ± 0.27 |
| (Test 2 - BERT) Top-K Buckets        | 0.85 ± 0.09 |
```

The `Table` object is Mellea-ready and can be used immediately with LLMs.
Let's just get it to work:
```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/document_mobject.py#L10-L24
from mellea.backends import ModelOption
from mellea import start_session

m = start_session()
for seed in [x*12 for x in range(5)]:
    table2 = m.transform(table1,
                         "Add a column 'Model' that extracts which model was used or 'None' if none.",
                         model_options={ModelOption.SEED: seed})
    if isinstance(table2, Table):
        print(table2.to_markdown())
        break
    else:
        print(f"==== TRYING AGAIN after non-useful output.====")
```
In this example, `table1` should be transformed to have an extra column `Model` which contains the model string from the `Feature` column or `None` if there is none. Iterating through some seed values, we try to find a version which returns a parsable representation of the table. If found, print it out.

The output for this code sample could be:
```markdown
table1=
| Feature                              | AUC         |
|--------------------------------------|-------------|
| Bag of Words                         | 0.63 ± 0.11 |
| (Test 1 - GPT-2) Average Probability | 0.71 ± 0.25 |
| (Test 2 - GPT-2) Top-K Buckets       | 0.87 ± 0.07 |
| (Test 1 - BERT) Average Probability  | 0.70 ± 0.27 |
| (Test 2 - BERT) Top-K Buckets        | 0.85 ± 0.09 |

===== 18:21:00-WARNING ======
added a tool message from transform to the context as well.

table2=
| Feature                              | AUC         | Model   |
|--------------------------------------|-------------|---------|
| Bag of Words                         | 0.63 ± 0.11 | None    |
| (Test 1 - GPT-2) Average Probability | 0.71 ± 0.25 | GPT-2   |
| (Test 2 - GPT-2) Top-K Buckets       | 0.87 ± 0.07 | GPT-2   |
| (Test 1 - BERT) Average Probability  | 0.70 ± 0.27 | BERT    |
| (Test 2 - BERT) Top-K Buckets        | 0.85 ± 0.09 | BERT    |
```

The model has done a great job at fulfilling the task and coming back with a parsable syntax. You could now call (e.g. `m.query(table2, "Are there any GPT models referenced?")`) or continue transformation (e.g. `m.transform(table2, "Transpose the table.")`).


### MObject methods are tools

When an object is `mified` all methods with a docstring get registered as tools for the LLM call. You can control if you only want a subset of these functions to be exposed by two parameters (`funcs_include` and `funcs_exclude`):
```python
from mellea.stdlib.components.mify import mify

@mify(funcs_include={"from_markdown"})
class MyDocumentLoader:
    def __init__(self) -> None:
        self.content = ""

    @classmethod
    def from_markdown(cls, text: str) -> "MyDocumentLoader":
        doc = MyDocumentLoader()
        # Your parsing functions here.
        doc.content = text
        return doc

    def do_hoops(self) -> str:
        return "hoop hoop"
```
Above, the `mified` class `MyDocumentLoader` only exposes the `from_markdown()` method as tool to the LLM.


Here is an example, how the methods are handled with an LLM call. Imagine the following two calls that should lead to the same result:
```python
table1_t = m.transform(table1, "Transpose the table.") # the LLM function
table1_t2 = table1.transpose() # the table method
```
Every native function of `Table` is automatically registered as a tool to the transform function. I.e., here the `.transform()` function calls the LLM and the LLM will get back suggesting to use the very own `.transpose()` function to achieve the result - it will also give you a friendly warning that you could directly use the function call instead of using the transform function.


## Chapter 6: Tuning Requirements and Components

One of the main principles of generative programming is that you should prompt models in the same way that the models were aligned. But sometimes off-the-shelf models are insufficient. Here are some scenarios we have encountered:

 * you are introducing a custom Component with non-trivial semantics that are not well-covered by any existing model's training data
 * off-shelf-models fail to recognize important business constraints
 * you have a proprietary labeled dataset which you would like to use for improving classification, intent detection, or another requirement-like task.

The third case is very common. In this tutorial we will explore a case-study focused on that case. we walk through fine-tuning a LoRA adapter using classification data to enhance a requirement checker. We then explain how this fine-tuned adapter can be incorporated into a Mellea program.

### Problem Statement

The Stembolt MFG Corporation we encountered in [Chapter 4](#chapter-4-generative-slots) is now is developing an AI agent to improve its operational efficiency and resilience. A key component of this pipeline is the AutoTriage module. AutoTriage is responsible for automatically mapping free-form defect reports into categories like mini-carburetor, piston, connecting rod, flywheel, piston rings, no_failure.

To ensure the generated output meets specific downstream system requirements, we require that each defect summary contains an identified failure mode. Unfortunately, LLMs perform poorly on this task out-of-the-box; stembolts are a niche device and detect reports are not commonly discussed on the open internet. Fortunately, over the years, Stembolt MFG has collected a large dataset mapping notes to part failures, and this is where the classifier trained via aLoRA comes in.

Here's peak at a small subset of Stembolt MFG's carefully [dataset of stembolt failure modes](https://github.com/generative-computing/mellea/blob/main/docs/examples/aLora/stembolt_failure_dataset.jsonl):

```json
{"item": "Observed black soot on intake. Seal seems compromised under thermal load.", "label": "piston rings"}
{"item": "Rotor misalignment caused torsion on connecting rod. High vibration at 3100 RPM.", "label": "connecting rod"}
{"item": "Combustion misfire traced to a cracked mini-carburetor flange.", "label": "mini-carburetor"}
{"item": "stembolt makes a whistling sound and does not complete the sealing process", "label": "no_failure"}
```

Notice that the last item is labeled "no_failure", because the root cause of that issue is user error. Stembolts are difficult to use and require specialized training; approximately 20% of reported failures are actually operator error. Classifying operator error as early in the process as possible -- and with sufficient accuracy -- is an important KPI for the customer service and repairs department of the Stembolt division.

Let's see how Stembolt MFG Corporation can use tuned LoRAs to implement the AutoTriage step in a larger Mellea application.

### Training the aLoRA Adapter

Mellea provides a command-line interface for training [LoRA](https://arxiv.org/abs/2106.09685) or [aLoRA](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md#activated-lora-alora) adapters.  Classical LoRAs must re-process our entire context, which can get expensive for quick checks happening within an inner loop (such as requirement checking). The aLoRA method allows us to adapt a base LLM to new tasks, and then run the adapter with minimal compute overhead. The adapters are fast to train and fast to switch between.

We will train a lightweight adapter with the `m alora train` command on this small dataset:

> [!NOTE]
> This script will require access to a gpu to run. You could also run this on your cpu, but it might take a while.
> For mac users, you might not be able to run this script as is, given the lack of `fp16` support in the accelerate library.

```bash
m alora train \
    --basemodel ibm-granite/granite-4.0-micro \
    --outfile stembolts_model \
    --adapter alora \
    docs/examples/aLora/stembolt_failure_dataset.jsonl
```

The default prompt format is `<|start_of_role|>check_requirement<|end_of_role|>`; this prompt should be appended to the context just before activated our newly trained aLoRA. If needed, you can customize this prompt using the `--promptfile` argument.

#### Parameters

While training adapters, you can easily tuning the hyper-parameters as below:

| Flag              | Type    | Default   | Description                                      |
|-------------------|---------|-----------|--------------------------------------------------|
| `--basemodel`     | `str`   | *required*| Hugging Face model ID or local path              |
| `--outfile`       | `str`   | *required*| Directory to save the adapter weights            |
| `--adapter`       | `str`   | `"alora"` | Choose between `alora` or standard `lora`        |
| `--epochs`        | `int`   | `6`       | Number of training epochs                        |
| `--learning-rate` | `float` | `6e-6`    | Learning rate                                    |
| `--batch-size`    | `int`   | `2`       | Per-device batch size                            |
| `--max-length`    | `int`   | `1024`    | Max tokenized input length                       |
| `--grad-accum`    | `int`   | `4`       | Gradient accumulation steps                      |
| `--promptfile`    | `str`   | None      | Directory to load the prompt format              |


### Upload to Hugging Face (Optional)

To share or reuse the trained adapter, use the `m alora upload` command to publish your trained adapter:

> [!IMPORTANT]
> Change `$HF_USERNAME` to your username, or else set that env var.
> Your model will be uploaded to Huggingface in a private repo.

```bash
m alora upload \
   --intrinsic \
   --name "$HF_USERNAME/stembolts" \
   --io-yaml io.yaml \
    stembolts_model
```

This will:
- Create the Hugging Face model repo (if it doesn't exist)
- Upload the contents of the `outfile` directory
- Requires a valid `HF_TOKEN` via `huggingface-cli login`

If you get a permissions error, make sure you are logged in to Huggingface:

```bash
huggingface-cli login  # Optional: only needed for uploads
```

> [!NOTE]
> **Warning on Privacy:** Before uploading your trained model to the Hugging Face Hub, review the visibility carefully. If you will be sharing your model with the public, consider whether your training data includes any proprietary, confidential, or sensitive information. Language models can unintentionally memorize details, and this problem compounds when operating over small or domain-specific datasets.


### Generating a README for Your Adapter

After training and uploading your adapter, you can use the `m alora add-readme` command to automatically generate a README for your HuggingFace model repository. The readme generator uses Mellea itself to analyze your training dataset and produce documentation that includes a description of the adapter, training data examples, and ready-to-use Python integration code.

```bash
m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml docs/examples/aLora/io.yaml \
    --basemodel granite-4.0-micro \
    docs/examples/aLora/stembolt_failure_dataset.jsonl
```

The generator will:
1. Load the first five samples from your JSONL dataset
2. Use an LLM to infer metadata: a high-level description, dataset summary, a CamelCase class name, and a Python function signature matching your input data structure
3. Validate the generated metadata using rejection sampling with deterministic checks (e.g., ensuring the arglist parses as valid Python)
4. Render a Jinja2 template into a complete README with HuggingFace-compatible YAML frontmatter, training data examples, and a full Python code snippet showing how to define an adapter class and use the intrinsic
5. Display the result and prompt you to confirm upload to HuggingFace

You can optionally provide a `--hints` file with domain-specific context to help the LLM produce more accurate descriptions:

```bash
m alora add-readme \
    --name $HF_USERNAME/stembolts \
    --io-yaml docs/examples/aLora/io.yaml \
    --basemodel granite-4.0-micro \
    --hints hints.txt
    docs/examples/aLora/stembolt_failure_dataset.jsonl
```

The readme generator can also be used programmatically:

```python
from cli.alora.readme_generator import generate_readme

generate_readme(
    dataset_path="stembolt_failure_dataset.jsonl",
    base_model="granite-4.0-micro",
    prompt_file=None,
    output_path="my_adapter_readme.md",
    name="your-username/stembolts-alora",
    hints=None,
)
```

> [!NOTE]
> The readme generator is itself a Mellea generative program. It uses `m.instruct` with a Pydantic output format, deterministic requirements, and `RejectionSamplingStrategy` to reliably generate structured metadata -- the same patterns described earlier in this tutorial.

### Integrating the Tuned Model into Mellea

After training an aLoRA classifier for our task, we would like to use that classifier to check requirements in a Mellea program. First, we need to setup our backend for using the aLoRA classifier:

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.adapters.adapter import CustomGraniteCommonAdapter

class StemboltAdapter(CustomGraniteCommonAdapter):
    def __init__(self, base_model_name:str="granite-4.0-micro"):
        super().__init__(
            model_id="$USERNAME/stembolts", # REPLACE $USERNAME WITH YOUR HUGGINGFACE USERNAME
            intrinsic_name="stembolts",
            base_model_name=base_model_name,
        )

backend = LocalHFBackend(
    model_id="ibm-granite/granite-4.0-micro", cache=SimpleLRUCache(5)
)

backend.add_adapter(StemboltAdapter(base_model_name="granite-4.0-micro"))
```

In the above arguments, `path_or_model_id` refers to the model checkpoint from last step, i.e., the `m alora train` process.

> [!NOTE]
> The `generation_prompt` passed to your `backend.add_alora` call should exactly match the prompt used for training.

There is a full example in [docs/examples/aLora/101_example.py](https://github.com/generative-computing/mellea/blob/main/docs/examples/aLora/101_example.py).

In this chapter, we have seen how a classification dataset can be used to tune a LoRA adapter on proprietary data. We then saw how the resulting model can be incorporated into a Mellea generative program. This is the tip of a very big iceberg.

## Chapter 7: On Context Management

Mellea manages context using two complementary mechanisms:

1. `Component`s themselves, which generally contain all of the context needed for a single-turn request. MObjects manage context using fields and methods, Instructions have a grounding_context for RAG-style requests, etc.

2. The `Context`, which stores and represents a (sometimes partial) history of all previous requests to the LLM made during the current session.

We have already seen a lot about how Components can be used to define the context of an LLM request, so in this chapter we will focus on the `Context` mechanism.

When you use the `start_session()` method, you are actually instantiating a `Mellea` with a default inference engine, a default model choice, and a default context manager. The following code is equivalent to `m.start_session()`:

```python
from mellea import MelleaSession

m = mellea.MelleaSession(
    backend=OllamaBackend(model_id=IBM_GRANITE_3_3_8B)
    context=SimpleContext()
)
```

The `SimpleContext` -- which is the only context we have used so far -- is a context manager that resets the chat message history on each model call. That is, the model's context is entirely determined by the current Component. Mellea also provides a `ChatContext`, which behaves like a chat history. We can use the ChatContext to interact with chat models:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/context_example.py#L1-L5
from mellea import start_session

m = mellea.start_session(ctx=ChatContext())
m.chat("Make up a math problem.")
m.chat("Solve your math problem.")
```

The `Context` object provides a few useful helpers for introspecting on the current model context; for example, you can always get the last model output:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/context_example.py#L7
print(m.ctx.last_output())
```

or the entire last turn (user query + assistant response):

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/context_example.py#L9
print(m.ctx.last_turn())
```

You can also use `session.clone()` to create a copy of a given session with its context at given point in time. This allows you to make multiple generation requests with the same objects in your context:
```python
m = start_session(ctx=ChatContext())
m.instruct("Multiply 2x2.")

m1 = m.clone()
m2 = m.clone()

# Need to run this code in an async event loop.
co1 = m1.ainstruct("Multiply that by 3")
co2 = m2.ainstruct("Multiply that by 5")

print(await co1)  # 12
print(await co2)  # 20
```
In the above example, both requests have `Multiply 2x2` and the LLM's response to that (presumably `4`) in their context. By cloning the session, the new requests both operate independently on that context to get the correct answers to 4 x 3 and 4 x 5.

## Chapter 8: Implementing Agents

> **Definition:**  An *agent* is a generative program in which an LLM determines the control flow of the program.

In the generative programs we have seen so far, the developer orchestrates a sequence of LLM calls. In contrast, agentic generative programs delegate control flow to the model itself. In this chapter we will see a couple of different ways of developing agents in Mellea:

1. **Classical Agents:** How to implement agentic loops in Mellea using the ReACT pattern.
2. **Guarded Nondeterminism:** We will return to the idea of generative slots, and see how this abstraction can help build more robust agents.

### Case Study: Implementing ReACT in Mellea

Let's build up to a full agent example using the ReACT pattern. We'll start with pseudocode and then incrementally build our Mellea ReACT program.

The core idea of ReACT is to alternate between reasoning ("Thought") and acting ("Action"):

```
# Pseudocode
while not done:
    get the model's next thought
    take an action based upon the thought
    choose arguments for the selected action
    observe the tool output
    check if a final answer can be obtained
return the final answer
```

Let's look at how this agent is implemented in Mellea:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/agents/react.py#L99
def react(
        m: mellea.MelleaSession,
        goal: str,
        react_toolbox: ReactToolbox,
        budget: int = 5,
):
    assert m.ctx.is_chat_context, "ReACT requires a chat context."
    test_ctx_lin = m.ctx.render_for_generation()
    assert (
            test_ctx_lin is not None and len(test_ctx_lin) == 0
    ), "ReACT expects a fresh context."

    # Construct the system prompt for ReACT.
    _sys_prompt = react_system_template.render(
        {"today": datetime.date.today(), "tools": react_toolbox.tools}
    )

    # Add the system prompt and the goal to the chat history.
    m.ctx.insert(mellea.stdlib.chat.Message(role="system", content=_sys_prompt))
    m.ctx.insert(mellea.stdlib.chat.Message(role="user", content=f"{goal}"))

    done = False
    turn_num = 0
    while not done:
        turn_num += 1
        print(f"## ReACT TURN NUMBER {turn_num}")

        print(f"### Thought")
        thought = m.chat(
            "What should you do next? Respond with a description of the next piece of information you need or the next action you need to take."
        )
        print(thought.content)

        print("### Action")
        act = m.chat(
            "Choose your next action. Respond with a nothing other than a tool name.",
            # model_options={mellea.backends.types.ModelOption.TOOLS: react_toolbox.tools_dict()},
            format=react_toolbox.tool_name_schema(),
        )
        selected_tool: ReactTool = react_toolbox.get_tool_from_schema(
            act.content)
        print(selected_tool.get_name())

        print(f"### Arguments for action")
        act_args = m.chat(
            "Choose arguments for the tool. Respond using JSON and include only the tool arguments in your response.",
            format=selected_tool.args_schema(),
        )
        print(
            f"```json\n{json.dumps(json.loads(act_args.content), indent=2)}\n```")

        # TODO: handle exceptions.
        print("### Observation")
        tool_output = react_toolbox.call_tool(selected_tool, act_args.content)
        m.ctx.insert(
            mellea.stdlib.chat.Message(role="tool", content=tool_output)
        )
        print(tool_output)

        is_done = IsDoneModel.model_validate_json(
            m.chat(
                f"Do you know the answer to the user's original query ({goal})? If so, respond with Yes. If you need to take more actions, then respond No.",
                format=IsDoneModel,
            ).content
        ).is_done
        if is_done:
            print("Done. Will summarize and return output now.")
            done = True
            return m.chat(
                f"Please provide your final answer to the original query ({goal})."
            ).content
        elif turn_num == budget:
            return None
```

### Guarded Nondeterminism

Recall Chapter 4, where we saw how libraries of `GenerativeSlot` components can be composed by introducing compositionality contracts. We will now build an "agentic" mechanism for automating the task of chaining together possibly-composable generative functions. Let's get started on our guarded nondeterminism agent ("guarded nondeterminism" is a bit of a mouthful, so we'll call this a a [Kripke](https://en.wikipedia.org/wiki/Saul_Kripke) agent going forward).

The first step is to add a new `Component` that adds preconditions and postconditions to generative slots:

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L10-L38 # TODO: MOVE THESE TO FAKE KRIPKE
class ConstrainedGenerativeSlot(Component):
    template = GEN_SLOT_TEMPLATE # the same template as is used for generative slots.

    def __init__(self, generative_slot: GenerativeSlot, preconds: list[Requirement | str], postconds: list[Requirement | str]):
        self._genslot = generative_slot
        self._preconds = [reqify(precond) for precond in preconds]
        self._postconds = [reqify(postcond) for postcond in postconds]

    def format_for_llm(self):
        return self._genslot.format_for_llm()

    def action_name(self):
        return self._genslot._function._function_dict["name"]
```

We'll also add a decorator for convienance:

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L41-L44
def constrained(preconds: list[Requirement | str], postconds: list[Requirement | str]):
    def _decorator(genslot: GenerativeSlot):
        return ConstrainedGenerativeSlot(genslot, preconds, postconds)
    return _decorator
```

We can now write down constrained generative slots like so:

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/main.py#L23-L27
@constrained(preconds=["contains a summary of the story's theme"], postconds=["each element of the list is the title and author of a significant novel"])
@generative
def suggest_novels_based_on_theme(summary: str) -> list[str]:
    """Based upon a summary of a short story, suggests novels with similar themes."""
    ...
```

Notice that we have used the `Requirement` component throughout, so we now have all the power of Mellea requirement validation semantics at our disposal for defining and checking pre/post-conditions.

We are now ready to provide the stump of our kripke agent:

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L54-L99
def filter_actions(m: mellea.MelleaSession, actions: list[ConstrainedGenerativeSlot], *, output: ModelOutputThunk | None = None):
  ...


def select_action(m: mellea.MelleaSession, actions: list[ConstrainedGenerativeSlot], goal: Requirement):
  ...


def kripke_agent(
        m: mellea.MelleaSession,
        actions: list[ConstrainedGenerativeSlot],
        goal: Requirement | str,
        budget: int = 10
) -> Callable[[str], str | None]:
    goal = reqify(goal)

    def _agent(initial_state: str) -> str | None:
        print(f"Goal: {goal.description}")
        m.ctx.insert(ModelOutputThunk(initial_state))
        i = 0
        while i in tqdm.tqdm(list(range(budget))):
            print(m.ctx.last_output())
            available_actions = filter_actions(m, actions)
            next_action = select_action(m, available_actions, goal)
            m.act(next_action)
            if goal.validate(m.backend, m.ctx):
                return m.ctx.last_output().value
        return None
    return _agent
```

The magic of the Kripke agent happens in `filter_actions`. The basic idea is simple: select only actions whose preconditions are implied by the current state:


```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L47-L55
def _check_action_preconditions(m: mellea.MelleaSession, action: ConstrainedGenerativeSlot, *, output: ModelOutputThunk | None = None) -> bool:
    for precondition in action._preconds:
        if not m.validate(precondition, output=output):
            return False
    return True


def filter_actions(m: mellea.MelleaSession, actions: list[ConstrainedGenerativeSlot], *, output: ModelOutputThunk | None = None):
    return [act for act in actions if _check_action_preconditions(m, act, output=output)]
```

And we finish of the agent by defining the selection criteria, using familiar constrained decoding techniques from our react agent:

```python
# file: https://github.com/generative-computing/kripke_agents/blob/main/kripke/base.py#L58-L71
def select_action(m: mellea.MelleaSession, actions: list[ConstrainedGenerativeSlot], goal: Requirement):
    # Setup a pydanyic model for the next action.
    action_names = [action.action_name() for action in actions]
    fields = dict()
    fields["next_action"] = Literal[*action_names]
    pydantic_model = pydantic.create_model("NextActionSelectionSchema", **fields)
    # Prompt the model for the next action.
    actions_list = "\n".join([f" * {action.action_name()}" for action in actions])
    action_selection_response = m.chat(f"Your ultimate goal is {goal.description}. Select the next action from the list of actions:\n{actions_list}", format=pydantic_model)
    # return the selected action.
    next_action_name = pydantic_model.model_validate_json(action_selection_response.content).next_action
    selected_action = [a for a in actions if a.action_name() == next_action_name]
    assert len(selected_action) == 1
    return selected_action[0]
```

We will stop here for the basic tutorial, but notice that there are several natural extensions:

1. We have not yet used the preconditions. Kripke agents can be optimized by **pre-computing** entailments between sets of pre-conditions and post-conditions; in this way, we only have to pay the cost of figuring out permissible interleaving of actions once.
2. We can execute multiple actions at once, then prune likely unfruitful portions of the search process.

We will dive into a full implementation of these and other Kripke agent tricks during a future deep-dive session on inference scaling with Mellea.

## Chapter 9: Interoperability with Other Frameworks

Mellea programs are, at last, just Python programs. Mellea programs can be shared via the Model Context Protocol or via the A2A protocol. Mellea programs can also consume tools and agents that implement these protocols.

### Simple mcp server running Mellea

Like we mentioned, mellea are at the end python programs. We can wrap a simple `mcp` server around a program and use the server as-is. Here is an example using [Pydantic AI's inbuild mcp server](https://ai.pydantic.dev/mcp/server/).

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/agents/mcp_example.py#L15-L40
# Create an MCP server
mcp = FastMCP("Demo")


@mcp.tool()
def write_a_poem(word_limit: int) -> str:
    """Write a poem with a word limit."""
    m = MelleaSession(OllamaModelBackend(model_ids.QWEN3_8B))
    wl_req = Requirement(
        f"Use only {word_limit} words.",
        validation_fn=simple_validate(lambda x: len(x.split(" ")) < word_limit),
    )

    res = m.instruct(
        "Write a poem",
        requirements=[wl_req],
        strategy=RejectionSamplingStrategy(loop_budget=4),
    )
    assert isinstance(res, ModelOutputThunk)
    return str(res.value)

if __name__ == '__main__':
    mcp.run()
```

### Running Mellea programs as an openai compatible server (Experimental)

We also provide an expiermental `m serve` utility for serving up an OpenAI-compatible **chat** endpoint. This allows you to write `m` programs that masquerade as a "model". To learn more about this functionality, run:

```shell
m serve --help
```

#### Example `m serve` application

While deploying programs using `m serve`, it is important for the programs to follow a specific structure. The programs needs a have function called `serve` with the following signature:

```python
# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/agents/m_serve_example.py#L25-L29
def serve(
    input: list[ChatMessage],
    model_options: None | dict = None,
    **kwargs
)
```

the `m serve` command then subsequently takes this function and runs a server that is openai compatible. For more information, please have a look at [this file](./examples/tutorial/m_serve_example.py) for how to write an `m serve` compatible program. To run the example:

```shell
m serve docs/examples/tutorial/m_serve_example.py
```

## Chapter 10: Prompt Engineering for M

Most backends operate on text. For these backends/models, Mellea has an opinionated stance on how to transform Python objects into text: the `TemplateFormatter`.

In most cases, you will want to create templates when adding a new component to the standard library or when customizing an existing component for a new model.

### Templates
Mellea's `TemplateFormatter` uses jinja2 templates to format objects when passing them to models for generation.

These templates can be stored directly in the class/object, or, more typically, the templates are stored in a directory, with each object having a specific file. For examples of the templates, see `mellea/templates/prompts/default`.

See the [customization section](#customization) below for a description of how the formatter chooses which template to use.

### Template Representations
Along with a template, each class/object needs to define the arguments that will be supplied when rendering the template. This happens in the component's `format_for_llm()` function. It returns either a string or a `TemplateRepresentation`.

`string`: the simplest approach is for this method to return a string representation of the object. This avoids templating altogether.

`TemplateRepresentation`: It can also return a `TemplateRepresentation` object.
This representation contains:
    - a reference to the component
    - a dictionary of arguments that will be passed to the template renderer
    - a list of tools/functions that relate to the component

It also contains either of the following fields
- template: a string representation of a jinja2 template that can be rendered with the provided args
- template_order: a list of strings describing the name of the template file to look up (without the ".jinja2" suffix); `*` denotes the class name.

### Customization
By writing a new template and/or changing the TemplateRepresentation of a component you can customize the textual representation. You can also customize based on the model.

#### Choosing a Template
Assuming a component's TemplateRepresentation contains a `template_order` field, the default TemplateFormatter grabs the relevant template by looking at the following places in order for each template in the `template_order`:
1. the formatter's cached templates if the template has been looked up recently
2. the formatter's specified template path
3. the package that the object getting formatted is from (either 'mellea' or some third party package)

If the default formatter searches the template path or the package, it uses the following logic:
- look in the `.../templates/prompts/...` directory
- traverse sub-directories in that path that match the formatter's model id (ie `ibm-granite/granite-4.0-h-micro` will match `.../templates/prompts/granite/granite-4-0-h/micro`) or default (ie `.../templates/prompts/default`)
- return the template at the deepest directory path
- the default template formatter assumes that a model will only have one match in any given directory; in other words, traversing a `templates` directory with both `prompts/granite/...` and `prompts/ibm/...` for `ibm-granite/granite-4.0-h-micro` should not happen

#### Editing an Existing Class
To customize the template and template representation of an existing class, simply create a new class that inherits from the class you want to edit. Then, override the format_for_llm function and create a new template.

See [`mellea/docs/examples/mify/rich_document_advanced.py`](./examples/mify/rich_document_advanced.py)

#### Decompose
For large prompts Mellea offers a decompose function to break down the initial prompt into a series of smaller ones.  In addition, the intermediary output allows for constraints and validation to be applied to the output of each step.  The final output is done by composing these intermediate steps.

See [`mellea/docs/examples/m_decompose/`](./examples/m_decompose/) for an input example prompt, intermediary json and python representations, and the final output.

For a smaller example see [`mellea/docs/examples/m_decompose/small/`](./examples/m_decompose/small/)

For an example within Python see [[`mellea/docs/examples/m_decompose/python/`](./examples/m_decompose/python/)]

To run the example locally use the following command:
```bash
m decompose run --out-dir ./ --prompt-file ./docs/examples/m_decomposeexample_decompose_prompt.txt
```

To run the generated intermediary use the following command:
```bash
python ./m_decomp_result.py > final_output.txt
```

## Chapter 11: Tool Calling
Mellea supports tool calling for providers/models that support it. Most session level functions support setting a tool_calls boolean. Setting this to true allows tools to be called, but there's no guarantee that a model will call them.

Tools can be made available for the model to call in a few ways:
1. Components: components can have a TemplateRepresentation object that contains tools.
2. Context: depending on the context, the components in that context can be used as sources of additional tools in the exact same way they would if they were the current action.
3. `ModelOptions.TOOLS`: model options can include a tools parameter. The preferred way of passing these tools is as a list of function objects.

Currently, tools are identified by the name of the function. If there are conflicts, the most recent tool with that name will be preferred. This means the tools available to the model will have the same priority listed above:
1. Tools from the current component will always be included
2. Tools from the context will be included if there are no name conflicts. A given context can decide what tools to surface, but in most cases, tools from the most recent component in the context will take priority over tools from older requests.
3. Tools from `ModelOptions.TOOLS` will only be added if they do not conflict with any of the above functions.

For examples on adding tools to the template representation of a component, see the `Table` object in [richdocument.py](../mellea/stdlib/docs/richdocument.py).

Here's an example of adding a tool through model options. This can be useful when you want to add a tool like web search that should almost always be available:
```python
import mellea
from mellea.backends import ModelOption, tool

@tool
def web_search(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
    """
    ...

m = mellea.start_session()
output = m.instruct(
    "Who is the 1st President of the United States?",
    model_options={
        ModelOption.TOOLS: [web_search],
    },
    tool_calls = True,
)

assert "web_search" in output.tool_calls

result = output.tool_calls["web_search"].call_func()
```

## Chapter 12: Asynchronicity
Mellea supports asynchronous behavior in several ways: asynchronous functions and asynchronous event loops in synchronous functions.

### Asynchronous Functions:
`MelleaSession`s have asynchronous functions that work just like regular async functions in python. These async session functions mirror their synchronous counterparts:
```python
m = start_session()
result = await m.ainstruct("Write your instruction here!")
```

However, if you want to run multiple async functions at the same time, you need to be careful with your context. By default, `MelleaSession`s use a `SimpleContext` that has no history. This will work just fine when running multiple async requests at once:
```python
m = start_session()
coroutines = []

for i in range(5):
    coroutines.append(m.ainstruct(f"Write a math problem using {i}"))

results = await asyncio.gather(*coroutines)
```

If you try to use a `ChatContext`, you will need to await between each request so that the context can be properly modified:
```python
m = start_session(ctx=ChatContext())

result = await m.ainstruct("Write a short fairy tale.")
print(result)

main_character = await m.ainstruct("Who is the main character of the previous fairy tail?")
print(main_character)
```

Otherwise, you're requests will use outdated contexts that don't have the messages you expect. For example,
```python
m = start_session(ctx=ChatContext())

co1 = m.ainstruct("Write a very long math problem.")  # Start first request.
co2 = m.ainstruct("Solve the math problem.")  # Start second request with an empty context.

results = await asyncio.gather(co1, co2)
for result in results:
    print(result)  # Neither request had anything in its context.

print(m.ctx)  # Only shows the operations from the second request.
```

Additionally, see [Chapter 7: Context Management](#chapter-7-on-context-management) for an example of how to use `session.clone()` to avoid these context issues.

### Asynchronicity in Synchronous Functions
Mellea utilizes asynchronicity internally. When you call `m.instruct`, you are using synchronous code that executes an asynchronous request to an LLM to generate the result. For a single request, this won't cause any differences in execution speed.

When using `SamplingStrategy`s or during validation, Mellea can speed up the execution time of your program by generating multiple results and validating those results against multiple requirements simultaneously. Whether you use `m.instruct` or the asynchronous `m.ainstruct`, Mellea will attempt to speed up your requests by dispatching those requests as quickly as possible and asynchronously awaiting the results.

## Appendix: Contributing to Mellea

### Contributor Guide: Getting Started

For detailed instructions on setting up your development environment, installing dependencies, configuring pre-commit hooks, and running tests, please see our **[Contributing Guide](../CONTRIBUTING.md#getting-started)**.

The Contributing Guide includes:
- Installation instructions (uv and conda)
- Development workflow and coding standards
- Testing guidelines and markers
- Troubleshooting common issues

### Contributor Guide: Requirements and Verifiers

Contributing new Requirements (i.e., verifiers) is an easy way to get started contributing to Mellea. Requirements can be as general or as domain-specific as you'd like, but must encapsulate a coherent and testable property. We have seen many examples of Requirements throughout this tutorial.

If you write a Requirement that is general-purose and likely useful to others, consider contributing your *general-purpose* component to Mellea's standard library:

1. Find a file in `mellea/stdlib/reqlib/` where your requirement belongs; if no file fits, create a new one.
2. Implement your requirement. Ideally, your verifier should be robust, which typically means not using the default LLMaJ behavior. If the requirement can be checked with code, you should write a validation function. See [our Markdown requirements](../mellea/stdlib/reqlib/md.py) for some examples of how this works. You could also [tune (and evaluate) a well-calibrated aLoRA](#chapter-6-tuning-requirements-and-components) for requirements that are not possible to implement in code.
3. Open a PR. If your Requirement uses LLMaJ, be sure to include a robust evaluation suite in your PR demonstrating that LLMaJ verification is good enough.

 One important note: if your requirement can be easily specified in terms of a grammatical constraint, then you should consider using constrained generation (by passing `format=` into your session or generate call -- see [the chapter on agent implementation](#chapter-8-implementing-agents) for some examples) instead of using requirements.

### Contributor Guide: Components
Components are the building blocks of Mellea. The point of a Component is that it has a way to represent itself to a Backend, its `format_for_llm` function. When creating a new component, you will most likely want to have `format_for_llm` return a `TemplateRepresentation`, a structured representation of itself that includes template args, tools, and the template itself.

Components are best created when you find yourself with data/objects that you are frequently formatting and marshalling into text to interact with LLMs.

To create a new component, you must both define it in code and (in most cases) create a template for it. Components are also runtime checkable protocols, so you need not inherit from the base class; you can simply add the required methods to an existing class as well.

When distributing a new Component, think of the Component the same way you think about a software library. Components are self-contained, well-documented, amenable to reuse, and hopefully also composable with other Components.

You have a couple of options for distributing your Component. You can distribute the Component as a library in user-space, or you can request that the Component is incorporated into the Mellea stdlib. Most Components are best positioned as third party libraries. You can distribute third-party generative programming components just like you distribute any third party library (github, pypi).

For Components that implement useful and widely used patterns, inclusion in the the Mellea stdlib may make sense. These are the early days of generative programming; we expect that some contributions will have pride-of-place in the Mellea standard library. We encourage contributors to ask early and often about inclusion in the stdlib.

### Contributor Guide: Specialized Mify
Mifying an object is another way to make it compatible with `Mellea`. Just like with Components, there is a `MifiedProtocol` that is a runtime checkable protocol. `@mify` or `mify(object)` adds the required methods to any object.

Since it's a protocol, you can create your own `mify` functions that wrap a class/object or add the required functionality to that class/object in any way you want.

For instance, you may have an ORM library where most of your objects follow the same pattern and structure. To integrate that library with `Mellea`, one approach would be to write a specific `mify` function that knows about that structure. It could look something like this:
```python
T = TypeVar("T")
def mify_orm(obj: T):
  setattr(obj, "format_for_llm", obj.sql)
  ...
```
In this way, you can define a common way to `mify` all components of this library on the fly, assuming they all have a `sql` function.

For a specialized mify function to be added to the stdlib, it must work as both a decorator and a function that can be called directly on objects/classes. It must also be a generic but useful pattern or a pattern for a widely used library.

### Contributor Guide: Sessions
While a less common need, Mellea allows you to create new types of sessions. When you need fine-grained control over context, it's advised that you completely override the `MelleaSession` methods.

To institute gates on calls that get made or modify calls without modifying the underlying context, overriding the methods but calling the `MelleaSession` supermethod is advised. See [the `chat-checker` example](./examples/sessions/creating_a_new_type_of_session.py).
