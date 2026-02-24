# pytest: skip_always

import mellea
from mellea.stdlib.components.docs.richdocument import RichDocument

m = mellea.start_session()

doc = RichDocument.from_document_file(
    "https://www.fs.usda.gov/foresthealth/docs/fidls/FIDL-78-ArmillariaRootDisease.pdf"
)

print(
    m.instruct(
        "Determine whether the mellea fungus only feeds on dead or weakened trees, thereby improving forest health",
        grounding_context={"usfs": doc},
    ).value
)
