from mellea.stdlib.tools import MelleaSearchTool
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.components import Instruction
import mellea


m = mellea.start_session()

# Call the search tool and get the parsed_repr of the too lresult.
tool = MelleaSearchTool()
results = tool.run("What is Mellea?")
results_repr = tool.parsed_repr(results)
results_repr = results_repr.with_contents()

# Call the model with the search results.
answer = m.instruct(
    "What programming language is Mellea written in?",
    grounding_context={"search_results": results_repr},
)


print(answer.value)
