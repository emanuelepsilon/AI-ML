# Agent Observability

Agentic systems should expose important intermediate steps. A useful trace records the user request, selected tools, retrieved documents, ranking scores, decisions, and final answer. This makes debugging possible when an answer is wrong or when a tool call was unnecessary.

For larger workflows, observability also helps compare agents. The manager can inspect which worker used which context, what assumptions were made, and whether the final answer depended on unsupported information.
