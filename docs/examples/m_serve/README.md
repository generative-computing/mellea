# M Serve Examples

This directory contains examples for deploying Mellea programs as API services using the `m serve` CLI command.

## Files

### m_serve_example_simple.py
A simple example showing how to structure a Mellea program for serving as an API.

**Key Features:**
- Defining a `serve()` function that takes input and returns output
- Using requirements and sampling strategies in served programs
- Custom validation functions for API constraints
- Handling chat message inputs

### pii_serve.py
Example of serving a PII (Personally Identifiable Information) detection service.

### client.py
Client code for testing the served API endpoints.

## Concepts Demonstrated

- **API Deployment**: Exposing Mellea programs as REST APIs
- **Input Handling**: Processing structured inputs (chat messages, requirements)
- **Output Formatting**: Returning appropriate response types
- **Validation in Production**: Using requirements in deployed services
- **Model Options**: Passing model configuration through API

## Basic Pattern

```python
from mellea import start_session
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import Requirement

session = start_session()

def serve(input: list[ChatMessage], 
          requirements: list[str] | None = None,
          model_options: dict | None = None):
    """Main serving function - called by m serve."""
    message = input[-1].content
    
    result = session.instruct(
        description=message,
        requirements=requirements or [],
        strategy=RejectionSamplingStrategy(loop_budget=3),
        model_options=model_options
    )
    return result
```

## Running the Server

```bash
# Start the server
m serve docs/examples/m_serve/m_serve_example_simple.py

# In another terminal, test with client
python docs/examples/m_serve/client.py
```

## API Endpoints

The `m serve` command automatically creates:
- `POST /generate`: Main generation endpoint
- `GET /health`: Health check endpoint
- `GET /docs`: API documentation (Swagger UI)

## Use Cases

- **Production Deployment**: Deploy Mellea programs as microservices
- **API Integration**: Integrate with existing systems via REST API
- **Scalability**: Run multiple instances behind a load balancer
- **Monitoring**: Add logging and metrics to served programs

## Related Documentation

- See `cli/serve/` for server implementation
- See `mellea/stdlib/session.py` for session management