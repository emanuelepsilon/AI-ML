"""
LlamaIndex Workflows - Step 1: Single-step workflow

Idea:
An agent can decide what to do dynamically.
A manual Workflow is different: you define the exact steps yourself.

This first workflow is the smallest possible version:

    StartEvent -> one @step function -> StopEvent

Meaning:
1. The workflow starts.
2. LlamaIndex sends a StartEvent into the first step.
3. The step does some work.
4. The step returns a StopEvent.
5. The workflow ends and gives back StopEvent.result.

No LLM.
No tools.
No RAG.

Just the workflow engine.
"""

import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


# ---------------------------------------------------------------------------
# 1. CREATE A WORKFLOW CLASS
# ---------------------------------------------------------------------------
# A workflow is a class that inherits from Workflow.
#
# Inside the class, functions decorated with @step become workflow steps.
class HelloBankWorkflow(Workflow):
    # -----------------------------------------------------------------------
    # 2. CREATE ONE STEP
    # -----------------------------------------------------------------------
    # @step tells LlamaIndex:
    # "This function is part of the workflow."
    #
    # ev: StartEvent means:
    # "This step starts when the workflow receives a StartEvent."
    #
    # -> StopEvent means:
    # "This step ends the workflow."
    @step
    async def greet_customer(self, ev: StartEvent) -> StopEvent:
        # This is the work done by the step.
        # In real workflows, this could be:
        # - validate input
        # - call an API
        # - retrieve documents
        # - call an LLM
        # - run a risk check
        message = "Hello! Welcome to the bank workflow."

        # StopEvent(result=...) ends the workflow and returns the final value.
        return StopEvent(result=message)


# ---------------------------------------------------------------------------
# 3. RUN THE WORKFLOW
# ---------------------------------------------------------------------------
# Workflows are async, so we use asyncio.run(...).
async def main():
    # Create the workflow object.
    workflow = HelloBankWorkflow(timeout=10, verbose=True)

    # Run the workflow.
    # Since our step takes StartEvent, LlamaIndex automatically starts there.
    result = await workflow.run()

    # result is the StopEvent.result value.
    print("\nWorkflow result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
