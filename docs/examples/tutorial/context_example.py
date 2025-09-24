from mellea import LegacyLinearContext, start_session

m = start_session(ctx=LegacyLinearContext())
m.chat("Make up a math problem.")
m.chat("Solve your math problem.")

print(m.ctx.last_output())

print(m.ctx.last_turn())
