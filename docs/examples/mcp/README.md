# Write a poem MCP
This is a simple example to show how to write a MCP tool 
with Mellea and instruct-validate-repair. Being able to 
speak the tool language allows you to integrate with
Claude Desktop, Langflow, ...

See code in [mcp_example.py](mcp_example.py)

## Run the example
You need to install the mcp package:
```bash
uv pip install "mcp[cli]"
```

and run the example in MCP debug UI:
```bash
uv run mcp dev docs/examples/tutorial/mcp_example.py
```


## Use in Langflow
Follow this path (JSON) to use it in Langflow: [https://docs.langflow.org/mcp-client#mcp-stdio-mode](https://docs.langflow.org/mcp-client#mcp-stdio-mode)

The JSON to register your MCP tool is the following. Be sure to insert the absolute path to the directory containing the mcp_example.py file:

```json
{
  "mcpServers": {
    "mellea_mcp_server": {
      "command": "uv",
      "args": [
        "--directory",
        "<ABSOLUTE PATH>/mellea/docs/examples/mcp",
        "run",
        "mcp",
        "run",
        "mcp_example.py"
      ]
    }
  }
}
```




