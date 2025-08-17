import pytest
from typing import Literal
from mellea import generative, session, instruct, chat, validate, query, transform
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.session import get_session, MelleaSession
from mellea.stdlib.mify import mify, MifiedProtocol
from mellea.stdlib.requirement import req


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]: ...


@generative
def generate_summary(text: str) -> str: ...


@mify(fields_include={"name", "age"})
class TestPerson:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def get_info(self) -> str:
        """Get person information."""
        return f"{self.name} is {self.age} years old"


def test_basic_contextual_session():
    """Test basic contextual session usage with convenience functions."""
    with session():
        # Test instruct
        result = instruct("Say hello")
        assert isinstance(result, ModelOutputThunk)
        assert result.value is not None
        
        # Test that we can get the session
        current_session = get_session()
        assert isinstance(current_session, MelleaSession)


def test_no_active_session_error():
    """Test error handling when no session is active."""
    with pytest.raises(RuntimeError, match="No active session found"):
        get_session()
    
    with pytest.raises(RuntimeError, match="No active session found"):
        instruct("test")
    
    with pytest.raises(RuntimeError, match="No active session found"):
        chat("test")


def test_generative_with_contextual_session():
    """Test generative slots work with contextual sessions."""
    with session():
        # Test without explicit session parameter
        result = classify_sentiment(text="I love this!")
        assert result in ["positive", "negative"]
        
        # Test with summary generation
        summary = generate_summary(text="A short text about something interesting.")
        assert isinstance(summary, str)
        assert len(summary) > 0


def test_generative_backward_compatibility():
    """Test that generative slots still work with explicit session parameter."""
    with session() as m:
        # Test old pattern still works
        result = classify_sentiment(m, text="I love this!")
        assert result in ["positive", "negative"]


def test_mify_with_contextual_session():
    """Test mify functionality with contextual sessions."""
    with session():
        person = TestPerson("Alice", 30)
        assert isinstance(person, MifiedProtocol)
        
        # Test query functionality
        query_result = query(person, "What is this person's name?")
        assert isinstance(query_result, ModelOutputThunk)
        
        # Test transform functionality  
        transform_result = transform(person, "Make this person 5 years older")
        # Transform can return either ModelOutputThunk or the tool output when tools are called
        assert transform_result is not None


def test_nested_sessions():
    """Test nested sessions behavior."""
    with session() as outer_session:
        outer_result = instruct("outer session test")
        assert isinstance(outer_result, ModelOutputThunk)
        
        with session() as inner_session:
            # Inner session should be active
            current_session = get_session()
            assert current_session is inner_session
            
            inner_result = instruct("inner session test")
            assert isinstance(inner_result, ModelOutputThunk)
        
        # After inner session exits, outer should be active again
        current_session = get_session()
        assert current_session is outer_session


def test_session_cleanup():
    """Test session cleanup after context exit."""
    session_ref = None
    with session() as m:
        session_ref = m
        instruct("test during session")
    
    # After exiting context, no session should be active
    with pytest.raises(RuntimeError, match="No active session found"):
        get_session()
    
    # Session should have been cleaned up
    assert hasattr(session_ref, 'ctx')


def test_all_convenience_functions():
    """Test all convenience functions work within contextual session."""
    with session():
        # Test instruct
        instruct_result = instruct("Generate a greeting")
        assert isinstance(instruct_result, ModelOutputThunk)
        
        # Test chat
        chat_result = chat("Hello there")
        assert hasattr(chat_result, 'content')
        
        # Test validate
        validation = validate([req("The response should be positive")])
        assert validation is not None
        
        # Test query with a mified object
        test_person = TestPerson("Test", 42)
        query_result = query(test_person, "What is the name?")
        assert isinstance(query_result, ModelOutputThunk)
        
        # Test transform with a mified object
        transform_result = transform(test_person, "Double the age")
        assert transform_result is not None


def test_session_with_parameters():
    """Test contextual session with custom parameters."""
    with session(backend_name="ollama", model_id="granite3.3:8b") as m:
        result = instruct("test with parameters")
        assert isinstance(result, ModelOutputThunk)
        assert isinstance(m, MelleaSession)


def test_multiple_sequential_sessions():
    """Test multiple sequential contextual sessions."""
    # First session
    with session():
        result1 = instruct("first session")
        assert isinstance(result1, ModelOutputThunk)
    
    # Ensure no session is active between contexts
    with pytest.raises(RuntimeError, match="No active session found"):
        get_session()
    
    # Second session
    with session():
        result2 = instruct("second session")
        assert isinstance(result2, ModelOutputThunk)


def test_contextual_session_with_mified_object_methods():
    """Test that mified objects work properly within contextual sessions."""
    with session():
        person = TestPerson("Bob", 25)
        
        # Test that mified object methods work
        query_obj = person.get_query_object("What's the age?")
        assert query_obj is not None
        
        transform_obj = person.get_transform_object("Make older")
        assert transform_obj is not None
        
        # Test format_for_llm
        llm_format = person.format_for_llm()
        assert llm_format is not None
        assert hasattr(llm_format, 'args')


def test_session_methods_with_mified_objects():
    """Test using session query/transform methods with mified objects."""
    with session() as m:
        person = TestPerson("Charlie", 35)
        
        # Test session query method 
        query_result = m.query(person, "What is this person's age?")
        assert isinstance(query_result, ModelOutputThunk)
        
        # Test session transform method  
        transform_result = m.transform(person, "Make this person younger")
        # Transform can return either ModelOutputThunk or tool output when tools are called
        assert transform_result is not None
        
        # Verify mified objects have query/transform object creation methods
        assert hasattr(person, 'get_query_object')
        assert hasattr(person, 'get_transform_object')
        assert hasattr(person, '_query_type')
        assert hasattr(person, '_transform_type')


if __name__ == "__main__":
    pytest.main([__file__])