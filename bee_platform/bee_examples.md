# Mellea-BeeAI

Mellea is a library for writing generative programs. 
BeeAI Framework is an open-source framework for building production-grade multi-agent systems.
This example serves to merge both libraries with a simple module that will allow users to transform
their Mellea programs into BeeAI agents.

We provide the example of an email writer.

# Initialization

First, install BeeAI, instructions available here: https://framework.beeai.dev/introduction/quickstart
Then, add the BeeAI-sdk to your local environment.
```bash
uv add beeai-sdk
```

# Running the example

Then, in order to run the example email writer, run:
```bash
uv run --with mellea docs/examples/bee_agent.py
```

In a separate terminal, either run
```bash
beeai run mellea_agent
```

OR open the UI and select the **mellea-agent**.

```bash
beeai ui
```

# Creating your own examples

To create your own BeeAI agent with Mellea, write a traditional program with Mellea. 

Ensure that the first parameter is an **m** object.

Wrap your Mellea program with ```@bee_app```.

Place your example in the ```docs/examples/``` folder.



