<p align="center" draggable="false"><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Module 8: Advanced Retrieval, Deep Agents, and Open Deep Research</h1>

> **Note:** This module covers three complementary topics: advanced retrieval strategies for RAG, building complex Deep Agents for long-horizon tasks, and implementing a supervisor-researcher architecture for automated deep research. All notebooks use a Stone Ridge Investment Advisory use case.

### Outline:

**Notebook 1: `01_Advanced_Retrieval_Assignment.ipynb`**

*Part 1: Retrieval Strategies*
- Task 1: Dependencies and API Keys
- Task 2: Data Collection and Preparation
- Task 3: Setting Up QDrant
- Task 4: Naive RAG Chain
- Task 5: BM25 Retriever
     - **Question #1**: BM25 vs embeddings comparison
- Task 6: Contextual Compression (Reranking)
- Task 7: Multi-Query Retriever
     - **Question #2**: Multi-query recall improvement
- Task 8: Parent Document Retriever
- Task 9: Ensemble Retriever
- Task 10: Semantic Chunking
     - **Question #3**: Semantic chunking behavior

*Part 2: Evaluation*
- **Activity #1**: Evaluate with Ragas

**Notebook 2: `02_Deep_Agents_Assignment.ipynb`**

*Breakout Room #1: Deep Agent Foundations*
- Task 1: Dependencies & Setup
- Task 2: Understanding Deep Agents
- Task 3: Planning with Todo Lists
- Task 4: Context Management with File Systems
- Task 5: Basic Deep Agent
     - **Question #1**: Todo list trade-offs
     - **Question #2**: Context management strategy
     - **Activity #1**: Build a Research Agent

*Breakout Room #2: Advanced Features & Integration*
- Task 6: Subagent Spawning
- Task 7: Long-term Memory Integration
- Task 8: Skills - On-Demand Capabilities
- Task 9: Using deepagents-cli
- Task 10: Building a Complete Deep Agent System
     - **Question #3**: Subagent configuration design
     - **Question #4**: Production investment advisory considerations
     - **Activity #2**: Build an Investment Advisory Agent

**Notebook 3: `03_Open_Deep_Research_Assignment.ipynb`**

*Breakout Room #1: Deep Research Foundations*
- Task 1: Dependencies
- Task 2: State Definitions
- Task 3: Utility Functions and Tools
- Task 4: Configuration System
- Task 5: Prompt Templates
     - **Question #1**: State interrelationships
     - **Question #2**: Imports vs inline code
     - **Activity #1**: Explore the Prompts

*Breakout Room #2: Building & Running the Researcher*
- Task 6: Node Functions
- Task 7: Graph Construction
- Task 8: Running the Deep Researcher
- Task 9: Understanding the Output
- Task 10: Key Takeaways & Next Steps
     - **Question #3**: Parallel vs sequential research trade-offs
     - **Question #4**: Production investment research application
     - **Activity #2**: Custom Investment Research

### Prerequisites:

#### 1. API Keys Required

You'll need the following API keys for this module:

- **OpenAI API Key**: For LLM, embeddings, and alternative models - [Get one here](https://platform.openai.com/api-keys)
- **Cohere API Key**: For reranking (Notebook 1) - [Get one here](https://dashboard.cohere.com/api-keys)
- **Anthropic API Key**: For Deep Agents and Deep Research (Notebooks 2 & 3) - [Get one here](https://console.anthropic.com/)
- **Tavily API Key**: For web search (Notebooks 2 & 3) - [Get one here](https://tavily.com/)
- **LangSmith API Key**: For tracing and debugging (optional) - [Get one here](https://smith.langchain.com/)

#### 2. Copy Environment Template

```bash
cp .env.sample .env
```
Then edit `.env` and replace the placeholder values with your actual API keys.

> NOTE: If you prefer to enter your API keys via prompts in the notebooks, you can skip this step.

### Steps to Run:

1. Install UV, which you can do through [this resource](https://docs.astral.sh/uv/#getting-started)
2. Run the command `uv sync`
3. Open your Jupyter notebook and select `.venv` for your kernel.

# Build

Run all three notebooks:
1. `01_Advanced_Retrieval_Assignment.ipynb` - Explore advanced retrieval strategies with the Alternative Investments Handbook
2. `02_Deep_Agents_Assignment.ipynb` - Build Deep Agents for investment advisory tasks
3. `03_Open_Deep_Research_Assignment.ipynb` - Run automated deep research on investment topics

# Ship

- Complete all questions and activities in all three notebooks
- Record a Loom video walking through the notebooks, the questions, and your implementations

# Share

- Make a social media post about your learnings and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

Here's a template to get your post started!

```
Just built advanced retrieval pipelines, Deep Agents, and an automated deep research system!

Key insights:
1. Different retrieval strategies have dramatic impacts on RAG quality
2. Deep Agents handle complex, multi-step tasks through planning, context management, and delegation
3. Supervisor-researcher architectures enable parallel, scalable research

Building better AI systems through better architecture!

A huge shoutout to @AI Makerspace for making this possible.

#AI #RAG #DeepAgents #LangChain #LangGraph #BuildInPublic
```

# Submitting Your Homework

## Main Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your repo:
    - To get the latest updates, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. Create a branch to track your changes. Example: `git checkout -b m08-assignment`
3. **IMPORTANT:** Start Cursor from the `08_Advanced_Retrieval_and_Deep_Research` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
4. Answer all Questions using the `##### Answer:` markdown cell below them in all three notebooks
5. Complete all Activities in all three notebooks
6. Add, commit and push your modified notebooks to your GitHub repository. _NOTE: Do not merge it into your main branch._

When submitting your homework, provide:
- Your Loom video link
- The GitHub URL to the `08_Advanced_Retrieval_and_Deep_Research` folder on your assignment branch
