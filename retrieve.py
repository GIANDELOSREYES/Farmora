from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# 1. Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to ChromaDB
vector_store = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

# 3. Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 4. Initialize the Ollama model
llm = ChatOllama(
    model="farming-guide",
    temperature=0.3,
    streaming = True
)

conversation_history = []

def rewrite_question(question, conversation_history):

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history
    )

    rewrite_prompt = f"""
    Given the conversation history and the latest user question,
    rewrite the latest question into a standalone question.

    If the question is already understandable without context,
    keep its original meaning.

    Do not answer the question.
    Only return the rewritten question.

    Conversation History:
    {history}

    Latest User Question:
    {question}

    Standalone Question:
    """

    response = llm.invoke(rewrite_prompt)

    return response.content.strip()


while True:
    # 5. Ask the user for a question
    question = input("Ask your farming question: ")

    # Rewrite the question using conversation history
    search_question = rewrite_question(question, conversation_history)

    print("\nRewritten Search Question:", search_question)
    # 6. Retrieve relevant information
    results = retriever.invoke(search_question)

    # 7. Combine retrieved chunks into one context
    context = "\n\n".join(
        document.page_content for document in results
    )

    # 8. Create a prompt
    prompt = f"""
    You are an Agriculture and Farming Guide chatbot.

    Your role is to provide helpful, practical, and understandable
    agricultural guidance to farmers.

    Use the agricultural information provided below as your
    primary reference when answering questions.

    STRICT RULES:

    1. Provide direct and natural answers. Do not start responses
       with phrases such as:
       - "According to the provided document..."
       - "Based on the provided information..."
       - "The agricultural information states..."
       
        1. Write as a knowledgeable and friendly farming assistant
            speaking directly to a farmer.

        2. NEVER mention the source of your information in normal
           answers.
        
        3. NEVER use these phrases:
           - "According to the provided information"
           - "According to the document"
           - "Based on the provided information"
           - "The document states"
           - "The agricultural information states"
           - "Based on the available information"
        
        4. Start your answer directly with the explanation,
           recommendation, or relevant information.
        
        5. Only discuss sources when the user explicitly asks
           where the information came from.
           
        RESPONSE STYLE EXAMPLES:

        Bad response:
        "According to the provided information, cover cropping
        helps improve soil quality."
        
        Good response:
        "Cover cropping helps improve soil quality by adding
        organic matter. It can also help control weeds and
        protect the soil."
        
        Bad response:
        "Based on the provided document, there is not enough
        information to answer your question."
        
        Good response:
        "I don't have enough information to give you an
        accurate answer to that question."

    2. Do not invent facts, statistics, fertilizer rates,
       pesticide dosages, or agricultural recommendations.

    3. Only provide factual agricultural information that is
       supported by the provided context.

    4. HANDLING UNKNOWN INFORMATION:

   - If the agricultural context does not contain the
     answer, respond naturally and briefly.
   - Never mention the document, provided information,
     retrieved context, or database.
   - Do not explain why the chatbot cannot find the answer.
   - Do not use phrases such as:
     "The provided information does not contain..."
     "According to the document..."
     "I must clarify that..."
     "Based on the available information..."
   - Do not invent an answer just to satisfy the user.
   - If necessary, briefly explain what additional
     information would be needed.

   Example:

   User: What is the exact nitrogen requirement of
   tomato plants in kilograms per hectare?

   Good response:
   "I don't have enough information to determine the
   exact nitrogen requirement of tomato plants."

   Bad response:
   "The provided agricultural information does not
   contain the exact nitrogen requirement of tomato
   plants in kilograms per hectare."

    5. You may use conversation history to understand follow-up
       questions, but do not treat previous AI-generated answers
       as verified agricultural facts.

    6. Explain agricultural concepts using simple and
       understandable language.

    7. Provide practical steps or examples when supported
       by the available information.

    8. Do not unnecessarily mention documents, sources, or
       retrieval processes unless the user specifically asks
       where the information came from.
       
    9. RESPONSE FORMATTING:

   For questions asking for explanations, benefits, causes,
   characteristics, examples, or recommendations:

   - Begin with a short introductory sentence.
   - Present the main information using bullet points.
   - Each bullet must contain one main idea.
   - Use bold text for important terms when appropriate.
   - Keep explanations concise and easy to understand.
   - Avoid long paragraphs.

   For questions asking for procedures or instructions:
   - Use numbered lists.
   - Present each step separately.

   For simple questions:
   - A short direct answer is acceptable.
   
   RESPONSE FORMAT EXAMPLE:

    Question: What are the benefits of cover cropping?
    
    Answer:
    
    Cover cropping provides several benefits:
    
    - **Improves soil quality:** Adds organic matter to the soil.
    - **Controls weeds:** Helps suppress weed growth.
    - **Protects the soil:** Covers exposed soil between planting periods.
    
    Always follow this general formatting style when appropriate.

    CONVERSATION HISTORY:
    {conversation_history}

    AGRICULTURAL INFORMATION:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
    """

    # 9. Generate an answer using Ollama
    print("\nFarming Guide: ", end="", flush=True)

    answer = ""

    for chunk in llm.stream(prompt):
        answer += chunk.content
        print(chunk.content, end="", flush=True)

    print()

    # Store the conversation
    conversation_history.append({
        "role": "user",
        "content": question
    })

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })