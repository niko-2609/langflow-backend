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
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

  # Add CORS middleware
app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # Configure appropriately for production
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # SSE client management
clients = defaultdict(asyncio.Queue)

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

async def emit_event(session_id: str, event_type: str, data: dict):
    """Emit an event to the SSE stream for a session."""
    if session_id in clients:
        event_data = {
            "type": event_type,
            "timestamp": asyncio.get_event_loop().time(),
            **data
         }
        await clients[session_id].put(json.dumps(event_data))

async def run_graph_with_streaming(session_id: str, graph, initial_state: dict, node_id_to_func_name: dict):
    """Run the graph with LangGraph's built-in streaming."""
    try:
        await emit_event(session_id, "workflow_start", {
            "message": "Workflow execution started",
            "total_steps": len(node_id_to_func_name)
        })

        step_count = 0
        final_result = None

        # Use LangGraph's built-in streaming
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            step_count += 1

            # Handle node execution updates
            for node_name, node_data in chunk.items():
                await emit_event(session_id, "node_update", {
                    "step": step_count,
                    "node": node_name,
                    "data": node_data,
                    "message": f"Completed node: {node_name}"
                })

                # Keep track of the final result
                final_result = node_data

        await emit_event(session_id, "workflow_complete", {
            "message": "Workflow completed successfully",
            "final_result": final_result
        })

    except Exception as e:
        await emit_event(session_id, "workflow_error", {
            "error": str(e),
            "message": "Workflow execution failed"
        })
    finally:
        # Signal completion and cleanup
        await emit_event(session_id, "stream_end", {
            "message": "Stream ended"
        })
        # Clean up client connection
        if session_id in clients:
            del clients[session_id]

class Node(BaseModel):
    id: str
    label: str
    nodeType: str

class Edge(BaseModel):
    source: str
    target: str

class WorkflowJson(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class WorkflowRequest(BaseModel):
    user_query: str
    workflowJson: WorkflowJson

class StreamWorkflowRequest(BaseModel):
    user_query: str
    workflowJson: WorkflowJson
    workflowId: Optional[str] = None

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
async def run_workflow_stream(req: WorkflowRequest, background_tasks: BackgroundTasks):
    """Start a streaming workflow execution."""
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    clients[session_id] = queue

    try:
        # Build the graph same as in /run-workflow
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

        # Compile the graph
        final_graph = graph_builder.compile()

        # Initial state
        initial_state = {
            "user_query": req.user_query,
            "llm_result": None,
            "is_coding_question": False,
            "accuracy_percentage": ""
        }

        # Start background streaming task
        background_tasks.add_task(run_graph_with_streaming, session_id, final_graph, initial_state, node_id_to_func_name)

    except Exception as e:
        # Clean up on error
        if session_id in clients:
            del clients[session_id]
        raise HTTPException(400, f"Error building workflow: {str(e)}")

    return {"session_id": session_id}

@app.post("/stream-workflow")
async def stream_workflow_direct(req: StreamWorkflowRequest):
    """Direct streaming endpoint that matches Next.js frontend expectations."""
    try:
        # Build the graph
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

        # Compile the graph
        final_graph = graph_builder.compile()

        # Initial state
        initial_state = {
            "user_query": req.user_query,
            "llm_result": None,
            "is_coding_question": False,
            "accuracy_percentage": ""
        }

        async def event_generator():
            try:
                # Send initial event
                yield {
                    "event": "workflow_start",
                    "data": json.dumps({
                        "type": "workflow_start",
                        "message": "Workflow execution started",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                }

                step_count = 0
                final_result = None

                # Use LangGraph's built-in streaming
                async for chunk in final_graph.astream(initial_state, stream_mode="updates"):
                    step_count += 1

                    # Handle node execution updates
                    for node_name, node_data in chunk.items():
                        event_data = {
                            "type": "step_update",
                            "step_id": str(step_count),
                            "step_name": node_name,
                            "status": "completed",
                            "data": node_data,
                            "message": f"Completed node: {node_name}",
                            "timestamp": asyncio.get_event_loop().time()
                        }

                        yield {
                            "event": "step_update",
                            "data": json.dumps(event_data)
                          }

                        # Keep track of the final result
                        final_result = node_data

                  # Send completion event
                yield {
                    "event": "workflow_complete",
                    "data": json.dumps({
                        "type": "workflow_complete",
                        "message": "Workflow completed successfully",
                        "final_result": final_result,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                }

            except Exception as e:
                # Send error event
                yield {
                    "event": "workflow_error",
                    "data": json.dumps({
                        "type": "workflow_error",
                        "error": str(e),
                        "message": "Workflow execution failed",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                }

        return EventSourceResponse(event_generator())

    except Exception as e:
        raise HTTPException(400, f"Error building workflow: {str(e)}")

@app.get("/stream/{session_id}")
async def stream(session_id: str):
    queue = clients.get(session_id)
    if not queue:
        raise HTTPException(404, "Session not found")

    async def event_generator():
        try:
            while True:
                # Wait for data with timeout to handle disconnections
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)

                    # Parse the JSON data to determine event type
                    event_data = json.loads(data)
                    event_type = event_data.get("type", "update")

                    yield {
                        "event": event_type,
                        "data": data,
                    }

                    # End stream if we receive stream_end event
                    if event_type == "stream_end":
                        break

                except asyncio.TimeoutError:
                    # Send keepalive to detect disconnections
                    yield {
                        "event": "keepalive",
                        "data": json.dumps({"type": "keepalive", "timestamp": asyncio.get_event_loop().time()}),
                    }

        except asyncio.CancelledError:
            print(f"Client disconnected from session {session_id}")
        finally:
            # Clean up session
            if session_id in clients:
                del clients[session_id]

    return EventSourceResponse(event_generator())