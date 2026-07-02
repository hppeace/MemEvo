
from pydantic import BaseModel

ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers.

# INSTRUCTIONS:
Use the provided memories to answer the question. Preserve names, numbers,
dates, places, frequencies, and specific entities exactly when available.

{context}

Question: {question}

Return a concise final answer grounded only in the context.
"""

def prepare_answer_prompt(memory_list: list[BaseModel], question: BaseModel) -> str:
    context_str = "\n".join(
        f"[{m.datatime}] {m.speaker} : {m.text}" for m in memory_list
    )
    return ANSWER_PROMPT.format(context=context_str, question=question)