import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly
based on the provided document context. If the answer isn't in the context,
say 'I could not find this in the provided documents.' Be concise and accurate."""

def generate_answer(question: str, context_chunks: list, chat_history: list) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = "\n\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from documents:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content