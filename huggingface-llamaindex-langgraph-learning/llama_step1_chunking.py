from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

document = Document(
    text=(
        "Alice has a mortgage. "
        "Ben has a personal loan. "
        "Alice has good credit. "
        "Ben has a high debt-to-income ratio. "
        "The bank requires manual review for risky applicants. "
        "Mortgage applicants need stable income and a property appraisal. "
        "Credit score and debt-to-income ratio are important risk signals."
    )
)

splitter = SentenceSplitter(
    chunk_size=30,
    chunk_overlap=5,
)

nodes = splitter.get_nodes_from_documents([document])

print(f"Number of nodes: {len(nodes)}\n")

for i, node in enumerate(nodes, start=1):
    print(f"Node {i}:")
    print(node.text)
    print()