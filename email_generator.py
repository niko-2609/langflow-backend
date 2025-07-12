from typing import TypedDict, Optional
from openai import OpenAI # type: ignore
from dotenv import load_dotenv 
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

from classifier import graph_builder # type: ignore


load_dotenv()

client  = OpenAI()

class State(TypedDict):
    name: str
    email: str
    goal: str
    tone: str
    email_body: str | None
    is_tone_correct: bool | None
    contact_id: str | None

class EmailResponse(BaseModel):
    email_body: str



class ToneCheckResponse(BaseModel):
    tone_match: bool



def collect_input(state: State) -> State:
    print("📥 InputCollectorNode")
    required_fields = ["name", "email", "goal", "tone"]
    for key in required_fields:
        if key not in state:
            raise ValueError(f"Missing field: {key}")
    return state




def generate_email(state: State) -> State:
    print("✉️ LLMEmailGeneratorNode")

    prompt = f'''
    You're an AI email assistant. Write a welcome email with its tone set to {state["tone"]}.

    User Name: {state["name"]}
    Goal: {state["goal"]}
    '''

    result = client.beta.chat.completions.parse(
        model="gpt-4.1",
        response_format=EmailResponse,
        messages=[
            {"role": "system", "content": "Generate a welcome email."},
            {"role": "user", "content": prompt}
        ]
    )

    state["email_body"] = result.choices[0].message.parsed.email_body if result.choices[0].message.parsed else None
    return state


def check_tone(state: State):
    print("🔍 ToneCheckerNode")

    prompt = f'''
    Is the following email written in a {state["tone"]} tone? 
    Check if the overall context of the email matches the tone mentioned. 

    {state["email_body"]}

    Return your answer as JSON: {{ "tone_match": true/false }}
    '''

    result = client.beta.chat.completions.parse(
        model="gpt-4.1",
        response_format=ToneCheckResponse,
        messages=[
            {"role": "system", "content": "Evaluate the tone of the email."},
            {"role": "user", "content": prompt}
        ]
    )

    result_tone = result.choices[0].message.parsed.tone_match if result.choices[0].message.parsed else None

    state["is_tone_correct"] = result_tone
    return state


def validate_tone(state: State):
    print("🔍 ValidateToneNode")
    if state["is_tone_correct"] != False:
        return "send"
    return "generate"

def save_contact(state: State) -> State:
    print("💾 UserSaverNode")
    # db = Prisma()
    # await db.connect()

    # contact = await db.contact.create({
    #     "email": state["email"],
    #     "name": state["name"],
    #     "customData": {
    #         "goal": state["goal"],
    #         "tone": state["tone"],
    #         "emailBody": state["email_body"],
    #     },
    #     # "createdAt": datetime.utcnow(),
    #     "flowId": state.get("flow_id"),
    #     "userId": state.get("user_id"),
    #     "webhookId": state.get("webhook_id")
    # })

    # await db.disconnect()
    # state["contact_id"] = contact.id
    return state




graph_builder = StateGraph(State)

graph_builder.add_node("collect_input", collect_input)
graph_builder.add_node("generate_email", generate_email)
graph_builder.add_node("check_tone", check_tone)
graph_builder.add_node("validate_tone", validate_tone)
graph_builder.add_node("save_contact", save_contact)

graph_builder.add_edge(START, "collect_input")
graph_builder.add_edge("collect_input", "generate_email")
graph_builder.add_edge("generate_email", "check_tone")
graph_builder.add_conditional_edges("check_tone", validate_tone, {
    "generate": "generate_email",
    "send": "save_contact"
})
graph_builder.add_edge("save_contact", END)


graph = graph_builder.compile()



def main():
    print("=== User Information Collection ===")
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    goal = input("Enter your email goal: ")
    tone = input("Enter desired tone (formal/casual/friendly): ")

    #initial state
    state = {
       "name": name,
       "email": email,
       "goal": goal,
       "tone": tone,
       "email_body": "",
       "is_tone_correct": False,
       "contact_id": ""

    }
    # Call the graph or pipeline
    graph_result = graph.invoke(state)

    print("graph_result", graph_result)



# Only run main() if this file is executed directly, NOT when imported
if __name__ == "__main__":
    main()
