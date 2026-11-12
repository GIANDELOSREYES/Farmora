import ollama
from httpcore import stream

while True:
    question = input("\nAsk your farming question: ")

    stream = ollama.chat(
        model='farming-guide',
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ],
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end="", flush=True)