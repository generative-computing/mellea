#!/usr/bin/env python3
"""
BeeAI Platform Integration Example

This example demonstrates how to use Mellea with the BeeAI Platform for GUI-based
chat interfaces and trace visualization.
"""

import os
from typing import Optional, Dict, Any
from mellea.stdlib.session import MelleaSession
from mellea.stdlib.base import CBlock
from mellea.stdlib.chat import Message
from mellea.backends.beeai_platform import BeeAIPlatformBackend
from mellea.backends.formatter import TemplateFormatter


def create_chat_agent(
    model_id: str = "granite3.3:2b",
    trace_granularity: str = "generate",
    enable_traces: bool = True,
) -> BeeAIPlatformBackend:
    """Create a BeeAI Platform backend configured for chat."""
    
    # Configure backend for local or remote usage
    base_url = os.getenv("BEEAI_BASE_URL", "http://localhost:11434")
    api_key = os.getenv("BEEAI_API_KEY")  # For OpenAI/Anthropic
    provider = os.getenv("BEEAI_PROVIDER", "ollama")  # Auto-detected for local models
    
    formatter = TemplateFormatter(model_id=model_id)
    
    backend = BeeAIPlatformBackend(
        model_id=model_id,
        formatter=formatter,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        trace_granularity=trace_granularity,
        enable_traces=enable_traces,
    )
    
    return backend


def chat_with_agent(
    message: str,
    session: Optional[MelleaSession] = None,
    model_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Chat with the agent and return response."""
    
    if session is None:
        backend = create_chat_agent()
        session = MelleaSession(backend=backend)
    
    # Create messages for the conversation
    system_message = Message(role="system", content="You are a helpful AI assistant created with Mellea and BeeAI Platform.")
    user_message = Message(role="user", content=message)
    
    # Add messages to session context
    session.ctx.add(system_message)
    session.ctx.add(user_message)
    
    # Generate response with tracing
    result = session.backend.generate_from_context(
        action=user_message,
        ctx=session.ctx,
        model_options=model_options or {
            "temperature": 0.7,
            "max_tokens": 512,
        }
    )
    
    return result.value


def serve(
    input: str,
    requirements: Optional[str] = None,
    model_options: Optional[Dict[str, Any]] = None,
) -> CBlock:
    """Serve function for OpenAI-compatible API (used by m serve command)."""
    
    # Extract message from input (could be string or list of messages)
    if isinstance(input, list) and len(input) > 0:
        # Input is list of messages from OpenAI format
        message = input[-1].get("content", str(input))
    else:
        message = str(input)
    
    # Generate response
    response = chat_with_agent(message, model_options=model_options)
    
    return CBlock(response)


def chat_demo():
    """Interactive chat demo for local testing."""
    
    print("🤖 BeeAI Platform Chat Demo")
    print("💡 This demo shows Mellea integration with BeeAI Platform")
    print("🔧 For GUI interface, run: m gui chat")
    print("="*50)
    
    # Create backend with tracing enabled
    backend = create_chat_agent(trace_granularity="all", enable_traces=True)
    session = MelleaSession(backend=backend)
    
    print(f"✅ Connected to {backend.model_id} via {backend.provider}")
    print(f"📊 Trace granularity: {backend.trace_granularity}")
    print(f"📁 Traces will be saved to: {backend.trace_output_dir}")
    print("\n💬 Start chatting (type 'quit' to exit, 'traces' to save traces):")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'traces':
                trace_file = backend.save_traces()
                print(f"💾 Traces saved to: {trace_file}")
                print(f"📊 Trace summary: {backend.get_trace_summary()}")
                continue
            elif user_input.lower() == 'clear':
                backend.clear_traces()
                print("🧹 Traces cleared")
                continue
            elif not user_input:
                continue
            
            # Generate response
            response = chat_with_agent(user_input, session=session)
            print(f"🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Save traces on exit
    if backend.enable_traces:
        trace_file = backend.save_traces()
        print(f"\n💾 Final traces saved to: {trace_file}")
        print(f"📊 Total traces collected: {len(backend.trace_context.traces)}")
    
    print("\n👋 Chat demo ended. Thanks for trying BeeAI Platform integration!")


def main():
    """Main function for direct execution."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            chat_demo()
        elif sys.argv[1] == "chat":
            # Single message chat
            message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello!"
            response = chat_with_agent(message)
            print(response)
        else:
            print("Usage: python beeai_platform_example.py [demo|chat <message>]")
    else:
        chat_demo()


if __name__ == "__main__":
    main()
