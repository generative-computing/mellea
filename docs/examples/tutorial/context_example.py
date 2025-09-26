from mellea import start_session
from mellea.stdlib.base import ChatContext

m = start_session(ctx=ChatContext())
m.chat("Make up a math problem.")
m.chat("Solve your math problem.")

print(m.ctx.last_output())
print("==================")
print(m.ctx.last_turn())
