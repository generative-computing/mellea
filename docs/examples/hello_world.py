# pytest: ollama, llm

import mellea

m = mellea.start_session()

email = m.instruct("Write an email inviting the interns to a lunch party.")

print(email)
