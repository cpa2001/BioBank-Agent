"""Streaming-first LLM client wrappers.

stream_parser.py implements an incremental tool-arguments JSON parser
so the runtime can render "about to call train_model with
icd10_code=E11..." while the LLM is still emitting tokens.

client.py wraps the existing biobank_agent.llm.LLMClient with an
``async_stream`` coroutine that emits AgentEvent deltas without blocking
the event loop.
"""
