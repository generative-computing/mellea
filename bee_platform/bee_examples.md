# Mellea-Agentstack

Mellea is a library for writing generative programs. 
Agentstack is an open-source framework for building production-grade multi-agent systems.
This example serves to merge both libraries with a simple module that will allow users to transform
their Mellea programs into Agentstack agentic interfaces with structured (form) inputs. 

We provide the example of an email writer. Only text inputs are currently supported.

# Installing Agentstack

To install Agentstack, follow the instructions at this page: https://agentstack.beeai.dev/introduction/welcome


# Running the example

Then, in order to run the example email writer, run:
```bash
uv run --with mellea --with agentstack-sdk docs/examples/agentstack.py
```

In a separate terminal, either run
```bash
agentstack run mellea_agent
```

OR open the UI and select the **mellea-agent**.

```bash
agentstack ui
```

# Tutorial

To create your own Agentstack agent with Mellea, write a traditional program with Mellea, as shown below:

```bash
def mellea_func(m: MelleaSession, sender: str, recipient, subject: str, topic: str, sampling_iters : int = 3) -> tuple[ModelOutputThunk, Context] | SamplingResult:
    """
    Example email writing module that utilizes an IVR loop in Mellea to generate an email with a specific list of requirements.
    Inputs:
        sender: str
        recipient: str
        subject: str
        topic: str
    Output:
        sampling: tuple[ModelOutputThunk, Context] | SamplingResult
    """
    requirements = [
        req("Be formal."),
        req("Be funny."),
        req(f"Make sure that the email is from {sender}, is towards {recipient}, has {subject} as the subject, and is focused on {topic} as a topic"),
        Requirement("Use less than 100 words.",
                   validation_fn=simple_validate(lambda o: len(o.split()) < 100))
    ]
    sampling = m.instruct(f"Write an email from {sender}. Subject of email is {subject}. Name of recipient is {recipient}. Topic of email should be {topic}.", requirements=requirements, strategy=RejectionSamplingStrategy(loop_budget=1), return_sampling_results=True)

    return sampling
```

As shown above, that the first parameter is an **m** object.

Wrap your Mellea program with ```@bee_app``` and title the file ```agentstack_agent.py```.

Place your example in the ```docs/examples/``` folder.



