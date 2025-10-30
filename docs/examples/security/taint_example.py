from mellea.stdlib.base import CBlock
from mellea.stdlib.session import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.security import privileged, SecurityError

# Create tainted content
tainted_desc = CBlock("Process this sensitive data")
tainted_desc.mark_tainted()

print(f"Original CBlock is tainted: {not tainted_desc.is_safe()}")

# Create session
session = MelleaSession(OllamaModelBackend("llama3.2"))

# Use tainted CBlock in session.instruct
print("Testing session.instruct with tainted CBlock...")
result = session.instruct(
    description=tainted_desc, 
)

# The result should be tainted
print(f"Result is tainted: {not result.is_safe()}")
if not result.is_safe():
    taint_source = result._meta['_security'].get_taint_source()
    print(f"Taint source: {taint_source}")
    print("✅ SUCCESS: Taint preserved!")
else:
    print("❌ FAIL: Result should be tainted but isn't!")

# Mock privileged function that requires safe input
@privileged
def process_safe_data(data: CBlock) -> str:
    """A function that requires safe (non-tainted) input."""
    return f"Processed: {data.value}"

print("\nTesting privileged function with tainted result...")
try:
    # This should raise a SecurityError
    processed = process_safe_data(result)
    print("❌ FAIL: Should have raised SecurityError!")
except SecurityError as e:
    print(f"✅ SUCCESS: SecurityError raised - {e}")