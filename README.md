# 💍 Multimodal Jewelry Personal Shopper Agent

An intelligent, multimodal AI agent built with **LangGraph**, **Gemini 2.5**, and **ColPali** built to support users through the jewelry purchasing journey. The agent maintains a sophisticated conversational state to infer key preferences, Style, Material, and Price, then leverages visual knowledge from product catalogs, and generates context-aware recommendations with real-time inventory availability

1. **Multimodal Visual Retrieval**: Uses **ColPali** (Vision Language Model) and **Qdrant** to retrieve and understand PDF product catalogs directly from visual queries, not only text.
2. **Stateful Conversation Management**: Built on **LangGraph** to handle complex, multi-turn dialogues, maintaining context across a session using persistent memory.
3. **Hierarchical Preference Inference**: Implements a structured inference chain, `Style` → `Material` → `Price` to systematically identify and refine user needs.
4. **Intelligent Guardrails**: Includes nodes for relevance checking, refusal of off-topic queries, and dynamic greeting handling.
5. **Visual Memory Sanitization**: Automatically captions and summarizes images in the chat history to maintain visual context while keeping the conversation concise and efficient.
6. **Smart Inventory Diagnosis**: If no exact matches are found, the agent performs a "diagnosis" step to suggest which filters such as price or material to relax for better results.

## 🛠️ Architecture

The project uses a graph-based architecture defined in `graph.py`. The workflow consists of the following key nodes:

1.  **Guardrails & Routing**:
    * `santizer`: Cleans conversation history and captions images using VLM.
    * `guardrail`: Classifies intent (Greeting, Related, or Off-topic).
    * `knowledge_router`: Decides if external knowledge retrieval is needed (RAG).

2.  **Visual RAG (Retrieval Augmented Generation)**:
    * `retrieve_documents`: Uses **ColPali** to search PDF catalogs for visual matches based on user queries.

3.  **Inference Engine**:
    * `infer_style`: Extracts style preferences (e.g., Solitaire, Halo).
    * `infer_material`: Extracts material preferences (e.g., Gold, Platinum).
    * `infer_price`: Extracts budget constraints and handles range logic.

4.  **Response Generation**:
    * `generate_final_response`: Synthesizes findings, performs inventory checks, and generates a visual response with product images.
    * `generate_no_preference`: Guides users who are undecided by showing a diverse "Smart Gallery".
    * `generate_conflict_response`: Handles impossible combinations or inventory gaps.

## 💻 Tech Stack

* **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/)
* **LLM**: Google Gemini 2.5 Flash
* **Vector Database**: [Qdrant](https://qdrant.tech/) [ChromaDB](https://www.trychroma.com/)
* **Visual Retrieval**: [ColPali](https://huggingface.co/vidore/colpali-v1.2) (By Vidore), [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
* **Backend**: FastAPI
* **Frontend**: Node.js 

## 📂 Project Structure

```bash
├── src
│   ├── configs.py           # Configuration for attribute extraction (Style/Material)
│   ├── config_nodes.py      # Node definitions
│   ├── state.py             # AgentState definition (TypedDict)
│   ├── graph.py             # Main LangGraph workflow definition
│   ├── utils.py             # Formats conversation history as text
│   ├── utils_db.py             # Checks availability and builds galleries in database
│   ├── vector_store.py          # ColPali + Qdrant integration code
│   └── nodes
│       ├── guardrails.py    # Relevance checks and safety
│       ├── memory.py        # Summarization and image captioning
│       ├── retrieve.py      # RAG logic
│       ├── generic_inference.py # Logic for attribute extraction
│       └── final_response.py    # Final answer generation
├── documents                # PDF Catalogs for RAG
└── requirements.txt
```

## 🔧 Setup & Installation
Clone the repository

```Bash
git clone https://github.com/LizyYu/Multimodal_Diamond_Shop_Agent.git
cd Jewellery_Agent
```

Run Qdrant (Docker) Ensure you have Docker installed and run a Qdrant instance:

```Bash
docker run -p 6333:6333 qdrant/qdrant
```

Start the web application

```Bash
uvicorn main:app --reload
npm run dev
```

## Usage Example
![usage example](./Jewellery_Agent/backend/web_page.png)

## 🧠 Logic Deep Dive
### ColPali Retrieval
Unlike traditional RAG that chunks text, this agent uses ColPali to embed entire PDF pages as visual vectors. This allows the agent to find rings based on "visual vibe" (e.g., "I want a ring that looks like a flower") even if the text description is sparse.

### Memory Sanitization
To prevent the context window from exploding with base64 image strings, the memory.py module uses a VLM to "watch" the conversation. When an image is passed, it replaces the heavy image data with a descriptive text caption (e.g., [User showed images of: A gold engagement ring with a halo setting]) for the long-term history.
