# RAG Reliability

A retrieval-augmented generation system should answer only when the retrieved evidence is strong enough. The assistant should inspect the top retrieved passages, decide whether they support the question, and cite the documents used. If the retrieval scores are weak or the sources do not contain the needed facts, the assistant should refuse or ask a clarifying question.

Reliable RAG systems separate answer quality from retrieval quality. A fluent answer is not enough. The system must show that the answer is supported by local evidence and that unsupported questions are handled conservatively.
