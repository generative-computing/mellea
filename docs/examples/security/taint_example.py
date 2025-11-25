from mellea.stdlib.base import CBlock
from mellea.stdlib.session import start_session
from mellea.security import SecLevel, privileged, SecurityError

# Create tainted content
tainted_desc = CBlock("Process this sensitive data", sec_level=SecLevel.tainted_by(None))

print(f"Original CBlock is tainted: {tainted_desc.sec_level.is_tainted() if tainted_desc.sec_level else False}")

# Create session
session = start_session()

# Use tainted CBlock in session.instruct
print("Testing session.instruct with tainted CBlock...")
result = session.instruct(
    description=tainted_desc, 
)

# The result should be tainted
print(f"Result is tainted: {result.sec_level.is_tainted() if result.sec_level else False}")
if result.sec_level and result.sec_level.is_tainted():
    taint_source = result.sec_level.get_taint_source()
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