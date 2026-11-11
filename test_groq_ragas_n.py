import sys
from unittest.mock import MagicMock

# Mock ChatVertexAI to prevent Ragas import error
module_path = "langchain_community.chat_models.vertexai"
dummy_module = type(sys)(module_path)
dummy_module.ChatVertexAI = MagicMock()
sys.modules[module_path] = dummy_module

import nest_asyncio
nest_asyncio.apply()

from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from ragas import evaluate

# Custom ChatOpenAI wrapper to handle n > 1 for Groq
class GroqChatOpenAI(ChatOpenAI):
    _force_n_to_one: bool = False

    @property
    def _default_params(self):
        params = super()._default_params
        if getattr(self, "_force_n_to_one", False):
            params["n"] = 1
        return params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", None) or getattr(self, "n", 1)
        if n <= 1:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        self._force_n_to_one = True
        orig_n = self.n
        self.n = 1
        
        generations = []
        try:
            for i in range(n):
                import time
                if i > 0:
                    time.sleep(0.5)
                res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
        finally:
            self._force_n_to_one = False
            self.n = orig_n
            
        from langchain_core.outputs import ChatResult
        return ChatResult(generations=generations, llm_output=res.llm_output)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", None) or getattr(self, "n", 1)
        if n <= 1:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        self._force_n_to_one = True
        orig_n = self.n
        self.n = 1
        
        generations = []
        try:
            for i in range(n):
                import asyncio
                if i > 0:
                    await asyncio.sleep(0.5)
                res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
        finally:
            self._force_n_to_one = False
            self.n = orig_n
            
        from langchain_core.outputs import ChatResult
        return ChatResult(generations=generations, llm_output=res.llm_output)

import os

API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print("Instantiating LLM and Embeddings...")
eval_llm = GroqChatOpenAI(
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=API_KEY)

ragas_llm = LangchainLLMWrapper(eval_llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

# Set LLM and Embeddings for metrics
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_emb

# Sample data
data = {
    "question": ["What is the capital of France?"],
    "contexts": [["Paris is the capital and most populous city of France."]],
    "answer": ["Paris is the capital of France."]
}

dataset = Dataset.from_dict(data)

print("Running Ragas evaluation with Groq...")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_emb
)
print("Ragas evaluation successful! Results:")
print(result)
