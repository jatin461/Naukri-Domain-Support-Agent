"""CrewAI wrapper around the grounded-generation service. Wired to the Retrieval Agent only."""
import json
from collections.abc import Callable

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.tools.recorder import InvocationRecorder


class PolicyQueryInput(BaseModel):
    query: str = Field(..., description="Policy question to answer from the knowledge base")


def result_to_json(result):
    return json.dumps(
        {
            "grounded": result.grounded,
            "answer": result.answer,
            "sources": result.source_doc_ids,
            "top_similarity": round(result.top_similarity, 4),
            "threshold": result.threshold,
        }
    )


class RagLookupTool(BaseTool):
    # The name deliberately contains "lookup": dispatch must follow the declared schema, not the name.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "rag_lookup"
    description: str = "Search the hiring-policy knowledge base and return a grounded answer with source documents."
    args_schema: type[BaseModel] = PolicyQueryInput
    search_fn: Callable = Field(exclude=True)  # query -> GroundedResult
    recorder: InvocationRecorder = Field(default_factory=InvocationRecorder, exclude=True)

    @property
    def invocations(self):
        return self.recorder.records

    def _run(self, query: str) -> str:
        result = self.search_fn(query)
        out = result_to_json(result)
        self.recorder.add(self.name, {"query": query}, json.loads(out))
        return out
