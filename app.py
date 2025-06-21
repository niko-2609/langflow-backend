# from fastapi import FastAPI, HTTPException
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
from fastapi import FastAPI, HTTPException
from collections import defaultdict


app = FastAPI()

from pydantic import BaseModel
from typing import List

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

    # Map UI label -> internal function name
    label_to_func_name = {
        "Classifier": "classify_message",
        "Router Query": "route_query",
        "General Query": "general_query",
        "Coding Query": "coding_query",
        "Coding Accuracy": "coding_query_accuracy",
    }

    # Map function name -> actual function
    func_name_to_function = {
        "classify_message": classify_message,
        "route_query": route_query,
        "general_query": general_query,
        "coding_query": coding_query,
        "coding_query_accuracy": coding_query_accuracy,
    }

    graph_builder = StateGraph(LangState)
    node_id_to_func_name = {}  # Save mapping of node.id -> func_name

    # Add all nodes to the graph
    for node in req.workflowJson.nodes:
        raw_label = node.label
        func_name = label_to_func_name.get(raw_label)
        if not func_name or func_name not in func_name_to_function:
            raise HTTPException(400, f"Unknown node label: {raw_label}")

        if func_name != "route_query":  # Don't register route_query
            graph_builder.add_node(func_name, func_name_to_function[func_name])
        node_id_to_func_name[node.id] = func_name

    # Add start edge
    start_nodes = [n for n in req.workflowJson.nodes if n.nodeType == 'start']
    if not start_nodes:
        raise HTTPException(400, "No start node found")
    start_id = start_nodes[0].id
    graph_builder.add_edge(START, node_id_to_func_name[start_id])

    # First, get all router node IDs
    router_node_ids = {n.id for n in req.workflowJson.nodes if n.nodeType == "router"}

    # Then, add direct edges only for non-router links
    for edge in req.workflowJson.edges:
        if edge.source in router_node_ids or edge.target in router_node_ids:
            continue  # Skip all edges connected to router nodes — handled later
        src_func = node_id_to_func_name[edge.source]
        tgt_func = node_id_to_func_name[edge.target]
          # Don't add edges *from* router functions — handled separately via add_conditional_edges
        if src_func == "route_query" or tgt_func == "route_query":
            continue
        graph_builder.add_edge(src_func, tgt_func)

    # Add END edges
    for node in req.workflowJson.nodes:
        if node.nodeType == 'end':
            graph_builder.add_edge(node_id_to_func_name[node.id], END)

    # Handle router nodes
    for node in req.workflowJson.nodes:
        if node.nodeType == 'router':
            router_id = node.id
            router_func_name = node_id_to_func_name[router_id]
            router_func = func_name_to_function["route_query"]

            # Find previous node to router
            parent_edges = [e for e in req.workflowJson.edges if e.target == router_id]
            if len(parent_edges) != 1:
                raise HTTPException(400, f"Router node '{router_func_name}' must have one incoming edge")
            parent_func = node_id_to_func_name[parent_edges[0].source]

            # Find child targets and assign conditions
            child_edges = [e for e in req.workflowJson.edges if e.source == router_id]
            if len(child_edges) < 1:
                raise HTTPException(400, f"Router node '{router_func_name}' must have at least one outgoing edge")

            condition_map = {}
            for e in child_edges:
                target_func = node_id_to_func_name[e.target]
                if "general" in target_func:
                    condition_map["general"] = target_func
                elif "coding" in target_func:
                    condition_map["coding"] = target_func
                else:
                    raise HTTPException(400, f"Could not infer route for {target_func}")

            graph_builder.add_conditional_edges(parent_func, router_func, condition_map)

    # Compile the graph (but don't run it)
    final_graph = graph_builder.compile()

    # # ✅ Print final graph structure
    # internal_graph = final_graph.get_graph()

    # print("📌 Final Graph Nodes:")
    # for node_name in internal_graph.nodes:
    #     print(f"  - {node_name}")

    # print("🔁 Final Graph Edges:")
    # # Robust handling: handle both dict or flat list of pairs
    # if isinstance(internal_graph.edges, dict):
    #     for src, targets in internal_graph.edges.items():
    #         print(f"  {src} -> {targets}")
    # else:
    #     edge_map = defaultdict(list)
    #     for pair in internal_graph.edges:
    #         if isinstance(pair, (list, tuple)) and len(pair) == 2:
    #             src, dst = pair
    #             edge_map[src].append(dst)
    #     for src, targets in edge_map.items():
    #         print(f"  {src} -> {targets}")

    state = {
        "user_query": req.user_query,
        "llm_result": None,
        "is_coding_question": False,
        "accuracy_percentage": ""
    }
    result = final_graph.invoke(state)
    return {"result": result}