from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from classifier import (
    classify_message,
    route_query,
    general_query,
    coding_query,
    coding_query_accuracy,
    State as LangState
)
from sse_starlette.sse import EventSourceResponse
import asyncio
import uuid
from collections import defaultdict

app = FastAPI()

# Configuration mappings
LABEL_TO_FUNC_NAME = {
    "Classifier": "classify_message",
    "Router Query": "route_query",
    "General Query": "general_query",
    "Coding Query": "coding_query",
    "Coding Accuracy": "coding_query_accuracy",
}

FUNC_NAME_TO_FUNCTION = {
    "classify_message": classify_message,
    "route_query": route_query,
    "general_query": general_query,
    "coding_query": coding_query,
    "coding_query_accuracy": coding_query_accuracy,
}

def add_workflow_nodes(graph_builder, nodes):
    """Add nodes to the graph and return node ID to function name mapping."""
    node_id_to_func_name = {}
    
    for node in nodes:
        func_name = LABEL_TO_FUNC_NAME.get(node.label)
        if not func_name or func_name not in FUNC_NAME_TO_FUNCTION:
            raise HTTPException(400, f"Unknown node label: {node.label}")
        
        if func_name != "route_query":
            graph_builder.add_node(func_name, FUNC_NAME_TO_FUNCTION[func_name])
        node_id_to_func_name[node.id] = func_name
    
    return node_id_to_func_name

def add_workflow_edges(graph_builder, edges, nodes, node_id_to_func_name):
    """Add edges to the graph, handling router nodes separately."""
    router_node_ids = {n.id for n in nodes if n.nodeType == "router"}
    
    # Add direct edges (non-router)
    for edge in edges:
        if edge.source in router_node_ids or edge.target in router_node_ids:
            continue
        
        src_func = node_id_to_func_name[edge.source]
        tgt_func = node_id_to_func_name[edge.target]
        
        if src_func != "route_query" and tgt_func != "route_query":
            graph_builder.add_edge(src_func, tgt_func)

def handle_router_nodes(graph_builder, nodes, edges, node_id_to_func_name):
    """Handle router node conditional edges."""
    for node in nodes:
        if node.nodeType != 'router':
            continue
            
        router_id = node.id
        
        # Find parent node
        parent_edges = [e for e in edges if e.target == router_id]
        if len(parent_edges) != 1:
            raise HTTPException(400, f"Router node must have one incoming edge")
        parent_func = node_id_to_func_name[parent_edges[0].source]
        
        # Find child targets and build condition map
        child_edges = [e for e in edges if e.source == router_id]
        if len(child_edges) < 1:
            raise HTTPException(400, f"Router node must have at least one outgoing edge")
        
        condition_map = {}
        for e in child_edges:
            target_func = node_id_to_func_name[e.target]
            if "general" in target_func:
                condition_map["general"] = target_func
            elif "coding" in target_func:
                condition_map["coding"] = target_func
            else:
                raise HTTPException(400, f"Could not infer route for {target_func}")
        
        graph_builder.add_conditional_edges(parent_func, FUNC_NAME_TO_FUNCTION["route_query"], condition_map)

class Node(BaseModel):
    id: str
    label: str
    nodeType: str  # Optional[str] if not always required

class Edge(BaseModel):
    source: str
    target: str

class WorkflowJson(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class WorkflowRequest(BaseModel):
    user_query: str
    workflowJson: WorkflowJson

@app.post("/run-workflow")
async def run_workflow(req: WorkflowRequest):
    print("✅ Parsed user_query:", req.user_query)
    print("✅ Parsed workflowJson:", req.workflowJson)

    graph_builder = StateGraph(LangState)
    
    # Add nodes and get mapping
    node_id_to_func_name = add_workflow_nodes(graph_builder, req.workflowJson.nodes)

    # Add start edge
    start_nodes = [n for n in req.workflowJson.nodes if n.nodeType == 'start']
    if not start_nodes:
        raise HTTPException(400, "No start node found")
    start_id = start_nodes[0].id
    graph_builder.add_edge(START, node_id_to_func_name[start_id])

    # Add edges
    add_workflow_edges(graph_builder, req.workflowJson.edges, req.workflowJson.nodes, node_id_to_func_name)

    # Add END edges
    for node in req.workflowJson.nodes:
        if node.nodeType == 'end':
            graph_builder.add_edge(node_id_to_func_name[node.id], END)

    # Handle router nodes
    handle_router_nodes(graph_builder, req.workflowJson.nodes, req.workflowJson.edges, node_id_to_func_name)

    # Compile the graph (but don't run it)
    final_graph = graph_builder.compile()
    state = {
        "user_query": req.user_query,
        "llm_result": None,
        "is_coding_question": False,
        "accuracy_percentage": ""
    }
    result = final_graph.invoke(state)
    return {"result": result}

@app.post("/run")
async def run_workflow(background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    clients[session_id] = queue

    background_tasks.add_task(fake_graph_runner, session_id)
    return {"session_id": session_id}




@app.get("/stream/{session_id}")
async def stream(session_id: str):
    queue = clients.get(session_id)
    if not queue:
        return {"error": "Session not found"}

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield {
                    "event": "node_update",
                    "data": data,
                }
        except asyncio.CancelledError:
            print("Client disconnected")
            del clients[session_id]

    return EventSourceResponse(event_generator())