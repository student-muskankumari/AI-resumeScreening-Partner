"""Term lists used by the rule engine.

All patterns are case-insensitive and bounded so that, for example, "ML"
does not match inside "HTML" and "agent" does not match "agentless".
Nothing here decides eligibility or points; it only names what to look for.
"""

from __future__ import annotations

import re

_LEFT = r"(?<![A-Za-z0-9])"
_RIGHT = r"(?![A-Za-z0-9])"


def rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile(_LEFT + "(?:" + "|".join(alternatives) + ")" + _RIGHT, re.I)


# --------------------------------------------------------------------------
# Phrases that mention AI but are NOT evidence of building AI systems:
# coding assistants and "AI-assisted development". They are removed from a
# line before any AI pattern is applied.
# --------------------------------------------------------------------------
AI_TOOL_PHRASES = rx(
    r"github copilot",
    r"copilot(?! studio)(?:[- ]class assistants?)?",
    r"cursor",
    r"claude code",
    r"claude ai",
    r"codex",
    r"chatgpt",
    r"ai[- ]assisted[a-z \-]{0,40}",
    r"ai[- ]powered full[- ]stack developer[a-z \-]{0,20}",
    r"ai (?:coding|development|dev)(?: tools?| assistants?)?",
    r"ai tools?",
    r"ai[- ]native[a-z \-]{0,25}",
)

# Generic marketing wording: counts as "AI was mentioned", never as evidence.
GENERIC_AI = rx(
    r"ai[- ](?:powered|driven|based|enabled|integrated)",
    r"artificial intelligence",
    r"ai models?",
    r"(?<!\.)ai",   # not the ".ai" of a domain name
)

# --------------------------------------------------------------------------
# Python stack
# --------------------------------------------------------------------------
PYTHON = {
    "Python": rx(r"python(?:\s?3(?:\.\d+)?)?"),
    "FastAPI": rx(r"fastapi"),
    "Django": rx(r"django(?: rest(?: framework)?)?", r"drf"),
    "Flask": rx(r"flask"),
    "Pandas": rx(r"pandas"),
    "NumPy": rx(r"numpy"),
    "PyTorch": rx(r"pytorch"),
    "Scikit-learn": rx(r"scikit[- ]?learn", r"sklearn"),
    "Streamlit": rx(r"streamlit"),
    "PySpark": rx(r"pyspark"),
    "Celery": rx(r"celery"),
    "SQLAlchemy": rx(r"sqlalchemy"),
    "pytest": rx(r"pytest"),
    "asyncio": rx(r"asyncio"),
    "Pydantic": rx(r"pydantic"),
    "Tkinter": rx(r"tkinter"),
}

# --------------------------------------------------------------------------
# LLM / RAG / agentic evidence (assignment section 3 examples and equivalents)
# --------------------------------------------------------------------------
GENAI = {
    "LLM": rx(r"llms?", r"large language models?", r"llmops", r"vlms?"),
    "RAG": rx(r"rag", r"retrieval[- ]augmented(?: generation)?", r"graph ?rag"),
    "LangChain": rx(r"langchain", r"lcel"),
    "LangGraph": rx(r"langgraph", r"langraph", r"stategraph"),
    "LlamaIndex": rx(r"llama[- ]?index", r"llamaparse"),
    "CrewAI": rx(r"crew ?ai"),
    "AutoGen": rx(r"autogen"),
    "Google ADK": rx(r"google adk", r"agent development kit"),
    "LangSmith": rx(r"langsmith"),
    "LiteLLM": rx(r"litellm"),
    "Haystack": rx(r"haystack"),
    "Spring AI": rx(r"spring ai"),
    "Vercel AI SDK": rx(r"vercel ai sdk"),
    "Claude Agent SDK": rx(r"claude agent sdk"),
    "AI agents": rx(
        r"agentic",
        r"multi[- ]agent",
        r"(?:ai|llm|voice|autonomous|specialist|specialized|supervisor|research|"
        r"review|website|trading|interview|news|langgraph|tool[- ]augmented)"
        r"(?: ai)?[- ]agents?",
        r"agents?(?: tool calls?| workflows?| orchestration| execution| memory)",
        r"\d+ (?:[a-z]+ )?agents",
        r"tool[- ]augmented",
    ),
    "Tool calling": rx(r"tool[- ]calling", r"function[- ]calling", r"tool calls?"),
    "MCP": rx(r"mcp", r"model context protocol"),
    "Embeddings": rx(r"embeddings?", r"sentence transformers"),
    "Vector search": rx(
        r"vector (?:db|dbs|database|databases|search|stores?|retrieval|index)",
        r"vdb",
        r"pinecone",
        r"chroma(?:db)?",
        r"weaviate",
        r"qdrant",
        r"milvus",
        r"faiss",
        r"pgvector",
        r"semantic (?:search|retrieval)",
        r"hybrid (?:search|retrieval)",
        r"rerank(?:ing|er)?",
    ),
    "OpenAI API": rx(r"openai", r"gpt[- ]?(?:3\.5|4o|4|5)[a-z0-9.\-]*", r"azure openai"),
    "Gemini": rx(r"gemini"),
    "Claude": rx(r"claude", r"anthropic", r"bedrock"),
    "Groq": rx(r"groq"),
    "Ollama": rx(r"ollama"),
    "Llama": rx(r"llama(?:[- ]?\d[a-z0-9.]*)?"),
    "Mistral": rx(r"mistral"),
    "Qwen": rx(r"qwen[a-z0-9.\-]*"),
    "Nvidia NIM": rx(r"nvidia nim"),
    "Vertex AI": rx(r"vertex ai"),
    "vLLM": rx(r"vllm"),
    "Prompt engineering": rx(
        r"prompt (?:engineering|design|orchestration|architectures?|construction|restructuring)",
        r"few[- ]shot",
        r"chain[- ]of[- ]thought",
    ),
    "Generative AI": rx(r"generative ai", r"gen ?ai"),
    "Fine-tuning": rx(r"fine[- ]tun(?:ing|ed|e)", r"q?lora", r"peft"),
    "NL-to-SQL": rx(
        r"nl[- ]to[- ]sql",
        r"text[- ]to[- ]sql",
        r"natural[- ]language (?:to )?(?:sql|soql)(?: analytics)?",
    ),
    "LLM evals": rx(r"ragas", r"evals", r"eval suite", r"llm (?:scoring|evaluation|judge)"),
    "Conversational AI": rx(r"conversational ai"),
}

# --------------------------------------------------------------------------
# Classical ML / deep learning / CV (AI exposure, but not LLM-agentic work)
# --------------------------------------------------------------------------
CLASSICAL_ML = {
    "Machine Learning": rx(r"machine learning", r"ml"),
    "Deep Learning": rx(r"deep learning", r"neural networks?", r"cnns?", r"convolutional", r"lstm", r"gru"),
    "TensorFlow": rx(r"tensor ?flow(?: lite)?", r"keras"),
    "PyTorch": rx(r"pytorch"),
    "Scikit-learn": rx(r"scikit[- ]?learn", r"sklearn"),
    "NLP": rx(r"nlp", r"natural language processing", r"sentiment (?:analysis|classification)", r"nltk", r"spacy"),
    "Transformers": rx(r"transformers?", r"bert", r"distilbert", r"roberta", r"bart", r"vit", r"hugging ?face"),
    "Computer Vision": rx(
        r"computer vision", r"opencv", r"yolo(?:v\d+)?(?:-face)?", r"object detection",
        r"image classification", r"segmentation", r"insightface", r"face recognition",
        r"mobilenet", r"resnet", r"efficientnet", r"ocr", r"tesseract",
    ),
    "Gradient boosting": rx(r"xgboost", r"lightgbm", r"random forest", r"logistic regression", r"svr", r"svm"),
    "ML modelling": rx(
        r"classification models?", r"predictive (?:models?|analysis|system)", r"prediction (?:pipeline|system|models?)",
        r"regressor", r"feature engineering", r"model (?:training|evaluation)", r"federated learning",
        r"meta[- ]heuristic", r"speech recognition", r"mlflow", r"mlops",
    ),
}

# --------------------------------------------------------------------------
# Backend, cloud, engineering depth
# --------------------------------------------------------------------------
FASTAPI = rx(r"fastapi")
OTHER_PY_FRAMEWORK = rx(r"django(?: rest(?: framework)?)?", r"flask")
ASYNC = rx(
    r"async(?:io|hronous(?:ly)?)?", r"await", r"concurren(?:t|cy)", r"celery",
    r"background (?:workers?|jobs?|tasks?|processing)", r"sse", r"server[- ]sent events",
    r"websockets?", r"streaming", r"event[- ]driven", r"thread[- ]safe", r"paralleli[sz]ed",
)
POSTGRESQL = rx(r"postgres(?:ql)?", r"pgvector", r"postgis")
OTHER_SQL = rx(r"mysql", r"sqlite", r"sql", r"mariadb", r"sql server", r"oracle db")
REDIS = rx(r"redis")

GCP = rx(r"gcp", r"google cloud(?: platform)?", r"cloud run", r"vertex ai", r"bigquery", r"gke")
OTHER_CLOUD = rx(
    r"aws", r"amazon web services", r"azure", r"ec2", r"s3", r"lambda", r"ecs", r"eks", r"fargate",
    r"sagemaker", r"digitalocean", r"heroku", r"vercel", r"netlify", r"railway", r"cloudflare",
    r"openstack", r"oci", r"onrender", r"deployed (?:on|to) render",
)
DOCKER = rx(r"docker(?:ized|izing| compose)?", r"containeri[sz](?:ed|ation|ing)")
DEPLOYMENT = rx(
    r"deploy(?:ed|ment|ments|ing|s)?", r"ci ?/ ?cd", r"ci-cd", r"github actions", r"gitlab ci",
    r"jenkins", r"kubernetes", r"k8s", r"terraform", r"argocd", r"nginx",
)
FRONTEND = rx(r"react(?:\.?js)?(?: native)?", r"next(?:\.?js)?", r"reactjs", r"nextjs")

ENGINEERING = {
    "testing": rx(
        r"pytest", r"unit tests?", r"unit testing", r"integration tests?", r"test suites?", r"test coverage",
        r"playwright", r"junit", r"jest", r"vitest", r"selenium", r"test cases", r"wiremock",
        r"automated test(?:s|ing)", r"e2e tests?", r"end-to-end test cases",
    ),
    "architecture": rx(
        r"microservices?", r"multi[- ]tenant", r"system design", r"architected", r"architecture",
        r"event[- ]driven", r"design patterns?", r"schema design", r"repository pattern",
    ),
    "caching": rx(r"cach(?:e|es|ed|ing)", r"lru"),
    "queues": rx(
        r"kafka", r"rabbitmq", r"celery", r"bullmq", r"sqs", r"queues?", r"pub[/-]sub",
        r"redis streams", r"qstash", r"activemq", r"airflow",
    ),
    "observability": rx(
        r"observability", r"monitoring", r"logging", r"prometheus", r"grafana", r"opentelemetry",
        r"tracing", r"kibana", r"sentry", r"cloudwatch", r"splunk", r"loki",
    ),
    "concurrency": rx(
        r"concurren(?:t|cy)", r"thread[- ]safe", r"multi[- ]?threading", r"asyncio", r"goroutines",
        r"semaphores?", r"paralleli[sz](?:ed|m)", r"async(?:hronous)?",
    ),
    "failure_handling": rx(
        r"retr(?:y|ies)", r"circuit breakers?", r"fallback", r"rate[- ]limit(?:ing|ed)?",
        r"idempoten(?:t|cy)", r"error (?:handling|recovery)", r"failover", r"fault tolerance",
        r"resilien(?:t|ce)", r"backoff", r"failure handling", r"graceful",
    ),
}

# --------------------------------------------------------------------------
# AI depth signals for the rule-based fallback. Each rule has groups; one
# group matched = strength 1, two or more = strength 2.
# --------------------------------------------------------------------------
AI_DEPTH_GROUPS = {
    "retrieval": [
        rx(r"rag", r"retrieval[- ]augmented(?: generation)?", r"graph ?rag"),
        rx(r"vector (?:db|dbs|database|databases|search|stores?|retrieval|index)", r"pinecone",
           r"chroma(?:db)?", r"weaviate", r"qdrant", r"milvus", r"faiss", r"pgvector",
           r"elasticsearch", r"opensearch"),
        rx(r"embeddings?"),
        rx(r"semantic (?:search|retrieval)", r"hybrid (?:search|retrieval)", r"rerank(?:ing|er)?",
           r"bm25", r"chunking", r"knowledge graph", r"context retrieval",
           r"retriev(?:er|al) (?:pipelines?|infrastructure|quality|optimization|layer)"),
    ],
    "agents_tools": [
        GENAI["AI agents"],
        rx(r"tool[- ]calling", r"function[- ]calling", r"tool calls?", r"tool integrations?",
           r"mcp", r"\d+-tool", r"tool[- ]augmented"),
        rx(r"supervisor", r"router", r"routing", r"orchestrat\w+", r"human[- ]in[- ]the[- ]loop"),
    ],
    "state_orchestration": [
        rx(r"langgraph", r"langraph", r"stategraph", r"state graph"),
        rx(r"stateful", r"state management", r"shared (?:graph )?state", r"checkpoint\w*", r"memory",
           r"session (?:persistence|continuity)", r"persistent \w+ state", r"thread_id"),
        rx(r"orchestrat\w+", r"multi[- ]step", r"multi[- ]stage", r"multi[- ]node", r"\d-stage",
           r"workflow (?:engine|orchestration|nodes?|graph)", r"(?:agentic|agent|llm) workflows?",
           r"human[- ]in[- ]the[- ]loop", r"conditional routing"),
    ],
    "data_pipeline": [
        rx(r"ingest(?:ion|ing)?"),
        rx(r"chunk\w*", r"pars(?:e|ing|er)", r"ocr", r"extraction", r"preprocess\w*", r"etl",
           r"document (?:processing|understanding)", r"scraping", r"crawling",
           r"feature engineering", r"augmentation", r"datasets?", r"corpus"),
        rx(r"(?:data|ingestion|processing|etl|nlp|ml|rag|inference|extraction|document|content|"
           r"embedding|training|rendering|retrieval|vision|voice|prediction|async|fastapi|llm|ai)"
           r"(?:[- ]\w+)?[- ]pipelines?",
           r"pipelines? (?:for|that|covering|integrating|processing|using|with)"),
    ],
    "evaluation": [
        rx(r"evals", r"eval suite", r"evaluation pipelines?", r"langsmith", r"ragas",
           r"llm (?:scoring|evaluation|judge)", r"qa framework", r"llm-to-llm"),
        rx(r"hallucinat\w+"),
        rx(r"benchmark\w*", r"\d+(?:\.\d+)?% (?:\w+ )?accuracy", r"accuracy (?:on|across)",
           r"guardrails", r"faithfulness", r"roc-auc", r"output (?:schema )?validation", r"quality checks?",
           r"structured output validation", r"validated against"),
    ],
}

# Rules where some groups are only supporting words. Without at least one of
# the listed core groups the rule scores nothing ("orchestration" alone is
# not evidence of agents).
AI_DEPTH_REQUIRED: dict[str, set[int]] = {"agents_tools": {0, 1}}

METRIC = re.compile(r"\d+(?:\.\d+)?\s?%|\d[\d,]*\+|\d+(?:\.\d+)?\s?(?:k|m)\b|sub-\d+", re.I)

# --------------------------------------------------------------------------
# Skills shown in the `matched_skills` output field (display name -> pattern).
# --------------------------------------------------------------------------
SKILLS: dict[str, re.Pattern[str]] = {
    "Python": PYTHON["Python"],
    "FastAPI": FASTAPI,
    "Django": PYTHON["Django"],
    "Flask": PYTHON["Flask"],
    "LangChain": GENAI["LangChain"],
    "LangGraph": GENAI["LangGraph"],
    "LlamaIndex": GENAI["LlamaIndex"],
    "CrewAI": GENAI["CrewAI"],
    "RAG": GENAI["RAG"],
    "LLM": GENAI["LLM"],
    "AI agents": GENAI["AI agents"],
    "MCP": GENAI["MCP"],
    "Embeddings": GENAI["Embeddings"],
    "Vector search": GENAI["Vector search"],
    "OpenAI API": GENAI["OpenAI API"],
    "Gemini": GENAI["Gemini"],
    "Ollama": GENAI["Ollama"],
    "Machine Learning": CLASSICAL_ML["Machine Learning"],
    "Deep Learning": CLASSICAL_ML["Deep Learning"],
    "TensorFlow": CLASSICAL_ML["TensorFlow"],
    "PyTorch": PYTHON["PyTorch"],
    "Scikit-learn": PYTHON["Scikit-learn"],
    "PostgreSQL": POSTGRESQL,
    "MySQL": rx(r"mysql"),
    "MongoDB": rx(r"mongo ?db"),
    "Redis": REDIS,
    "Kafka": rx(r"kafka"),
    "Docker": DOCKER,
    "Kubernetes": rx(r"kubernetes", r"k8s"),
    "GCP": GCP,
    "AWS": rx(r"aws", r"amazon web services"),
    "Azure": rx(r"azure"),
    "CI/CD": rx(r"ci ?/ ?cd", r"github actions", r"jenkins", r"gitlab ci"),
    "React": rx(r"react(?:\.?js)?", r"reactjs"),
    "Next.js": rx(r"next\.?js"),
    "Node.js": rx(r"node(?:\.?js)?", r"express(?:\.?js)?", r"nestjs"),
    "JavaScript": rx(r"javascript"),
    "TypeScript": rx(r"typescript"),
    "Java": rx(r"java"),
    "Spring Boot": rx(r"spring ?boot"),
    "Go": rx(r"golang", r"go(?=,| and | \()"),
    "C++": re.compile(r"(?<![A-Za-z0-9])c\+\+", re.I),
    "SQL": rx(r"sql"),
}
