"""Single-step LlamaIndex workflow example."""

import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class HelloBankWorkflow(Workflow):
    @step
    async def greet_customer(self, ev: StartEvent) -> StopEvent:

        message = "Hello! Welcome to the bank workflow."

        return StopEvent(result=message)


async def main():

    workflow = HelloBankWorkflow(timeout=10, verbose=True)

    result = await workflow.run()

    print("\nWorkflow result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
