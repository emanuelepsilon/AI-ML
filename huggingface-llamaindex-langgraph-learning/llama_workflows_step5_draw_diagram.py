"""LlamaIndex workflow visualization example."""

from pathlib import Path

from llama_index.core.workflow.drawing import draw_all_possible_flows

from llama_workflows_step3_branches import BranchingRiskWorkflow
from llama_workflows_step4_context_state import ContextStateRiskWorkflow

OUTPUT_DIR = Path("workflow_diagrams")
OUTPUT_DIR.mkdir(exist_ok=True)

branching_workflow = BranchingRiskWorkflow(timeout=10)
context_state_workflow = ContextStateRiskWorkflow(timeout=10)

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

print("Workflow diagrams created:")
print(branching_diagram.resolve())
print(context_state_diagram.resolve())
