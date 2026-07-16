"""
LlamaIndex Workflows - Step 5: Draw workflow diagrams

Idea:
When workflows get bigger, reading code is not always the easiest way to see
the flow.

LlamaIndex can draw an HTML graph showing:

    Event -> step -> Event -> step -> StopEvent

This is useful for:
    - debugging
    - documentation
    - explaining workflows to other people
    - checking branches visually

This script draws diagrams for workflows we already created:

    1. BranchingRiskWorkflow
    2. ContextStateRiskWorkflow
"""

from pathlib import Path

from llama_index.core.workflow.drawing import draw_all_possible_flows

from llama_workflows_step3_branches import BranchingRiskWorkflow
from llama_workflows_step4_context_state import ContextStateRiskWorkflow


# ---------------------------------------------------------------------------
# 1. OUTPUT FOLDER
# ---------------------------------------------------------------------------
# The HTML files will be written into this folder.
OUTPUT_DIR = Path("workflow_diagrams")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. CREATE WORKFLOW OBJECTS
# ---------------------------------------------------------------------------
# We do not need to run the workflows.
#
# draw_all_possible_flows(...) inspects the workflow class/steps and draws
# every possible route based on event types.
branching_workflow = BranchingRiskWorkflow(timeout=10)
context_state_workflow = ContextStateRiskWorkflow(timeout=10)


# ---------------------------------------------------------------------------
# 3. DRAW DIAGRAMS
# ---------------------------------------------------------------------------
# These create standalone HTML files.
#
# You can open them in a browser.
branching_diagram = OUTPUT_DIR / "branching_risk_workflow.html"
context_state_diagram = OUTPUT_DIR / "context_state_risk_workflow.html"

draw_all_possible_flows(
    branching_workflow,
    filename=str(branching_diagram),
)

draw_all_possible_flows(
    context_state_workflow,
    filename=str(context_state_diagram),
)


# ---------------------------------------------------------------------------
# 4. PRINT WHERE THEY ARE
# ---------------------------------------------------------------------------
print("Workflow diagrams created:")
print(branching_diagram.resolve())
print(context_state_diagram.resolve())
