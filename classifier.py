from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END # type: ignore
from openai import OpenAI # type: ignore
from dotenv import load_dotenv 
from pydantic import BaseModel


load_dotenv()

client  = OpenAI()

class State(TypedDict):
    user_query: str
    llm_result: str | None
    accuracy_percentage: str | None
    is_coding_question: bool | None

class ClassifyQueryResponse(BaseModel):
    is_coding_question: bool


class CodingAccuracyResponse(BaseModel):
    coding_query_accuracy: str

    
def classify_message(state: State):
    print("⚠️ classifier node")
    query = state["user_query"]


    SYSTEM_PROMPT = """ 
    You're an AI assitant. You should detect whether the user's query is a coding query or not.
    Return the response in specified JSON boolean only.
    """

    chat_result = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=ClassifyQueryResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            ]
    )

    result = chat_result.choices[0].message.parsed.is_coding_question if chat_result.choices[0].message.parsed else None

    print("Classifier result:",  result)

    state["is_coding_question"] = result

    return state

def route_query(state: State):
    print("⚠️ router node")

    if state["is_coding_question"]:
        return "coding"  # ➤ Matches the key in add_conditional_edges()
    return "general"



def general_query(state: State):
    print("⚠️ general query node")

    query = state["user_query"]

    SYSTEM_PROMPT = """
        You're an AI Assistant. Your job is to answer the user query as perfectly as possible.
    """

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )


    response = result.choices[0].message.content

    state["llm_result"] = response

    return state

def coding_query(state: State):
    print("⚠️ coding query node") 

    query = state["user_query"]

    SYSTEM_PROMPT= """
    You're an AI assistant expert in coding. You're job is to provide an accurate and efficient response for the user query
"""
    result = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    response = result.choices[0].message.content
    state["llm_result"] = response

    return state


def coding_query_accuracy(state: State):
    print("⚠️ coding accuracy node") 
    
    llm_response = state["llm_result"]

    SYSTEM_PROMPT = """
    You're an expert coding evaluating system. You're job is to judge and evaluate the content and give the accuracy of the code.
    Return the response in specified JSON format only.
"""
    if llm_response:
        result = client.beta.chat.completions.parse(
            model="gpt-4.1",
            response_format=CodingAccuracyResponse,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(llm_response)}
            ]
)
    
    response = result.choices[0].message.parsed.coding_query_accuracy if result.choices[0].message.parsed else None

    state["accuracy_percentage"] = response
    return state




graph_builder = StateGraph(State)

graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("route_query", route_query)
graph_builder.add_node("general_query", general_query)
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_query_accuracy", coding_query_accuracy)



graph_builder.add_edge(START, "classify_message")
graph_builder.add_conditional_edges("classify_message", route_query, {
    "general": "general_query",
    "coding": "coding_query"
})

graph_builder.add_edge("general_query", END)
graph_builder.add_edge("coding_query", "coding_query_accuracy")
graph_builder.add_edge("coding_query_accuracy", END)


final_graph = graph_builder.compile()


def main():
    userQuery = input("> ")

    #initial state
    state = {
       "user_query": userQuery,
       "llm_result": None,
       "is_coding_question": False,
       "accuracy_percentage": ""

    }
    # Call the graph or pipeline
    graph_result = final_graph.invoke(state)

    print("graph_result", graph_result)


# Only run main() if this file is executed directly, NOT when imported
if __name__ == "__main__":
    main()