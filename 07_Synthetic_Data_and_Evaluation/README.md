<p align="center" draggable="false"><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Module 7: Synthetic Data Generation and Evaluation with Ragas</h1>

> **Note:** This module brings together synthetic data generation and systematic evaluation into a single end-to-end workflow. You'll generate test data, build a RAG pipeline, evaluate it with Ragas metrics, iterate on the pipeline, and then explore agent evaluation.

### Outline:

**Notebook 1: `Synthetic_Data_Generation_and_RAG_Evaluation_Assignment.ipynb`**

*Part 1: Synthetic Data Generation*
- Task 1: Dependencies and API Keys
- Task 2: Data Preparation
- Task 3: Knowledge Graph Construction
- Task 4: Generating Synthetic Test Data
     - **Question #1**: Query synthesizer types
     - **Question #2**: Unrolled vs. abstracted SDG trade-offs
     - **Activity #1**: Custom Query Distribution

*Part 2: RAG Evaluation with Ragas*
- Task 5: Building a Baseline RAG Application
     - **Question #3**: Purpose of `chunk_overlap`
- Task 6: Evaluating with Ragas
- Task 7: Making Adjustments and Re-Evaluating
     - **Question #4**: Comparing system performance
     - **Question #5**: Benefits and limitations of synthetic data
     - **Question #6**: Production-critical metrics
     - **Activity #2**: Implement a Different Reranking Strategy

**Notebook 2: `Evaluating_Agents_Assignment.ipynb`**
- Task 1: Installing Required Libraries
- Task 2: Set Environment Variables
- Task 3: Building a ReAct Agent with Metal Price Tool
- Task 4: Implementing the Agent Graph Structure
- Task 5: Converting Agent Messages to Ragas Evaluation Format
- Task 6: Evaluating the Agent's Performance using Ragas Metrics
     - Tool Call Accuracy
     - Agent Goal Accuracy
     - Topic Adherence
     - **Question #1**: What is a "trace"?
     - **Question #2**: How metrics are calculated
     - **Question #3**: Metric implications for trust and safety
     - **Question #4**: Designing a comprehensive test suite
     - **Activity #1**: Evaluate Tool Call Accuracy
     - **Activity #2**: Evaluate Topic Adherence

### Prerequisites:

#### 1. API Keys Required

You'll need the following API keys for this module:

- **OpenAI API Key**: For LLM and embedding models - [Get one here](https://platform.openai.com/api-keys)
- **Cohere API Key**: For reranking models (Notebook 1) - [Get one here](https://dashboard.cohere.com/api-keys)
- **Metal API Key**: For the agent notebook (Notebook 2) - [Get one here](https://metals.dev/)

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

Run both notebooks:
1. `Synthetic_Data_Generation_and_RAG_Evaluation_Assignment.ipynb` - Generate synthetic test data and evaluate a RAG pipeline with Ragas
2. `Evaluating_Agents_Assignment.ipynb` - Evaluate agent behavior with Ragas metrics

# Ship

- Complete all questions and activities in both notebooks
- Make a diagram showing how Ragas metrics relate to RAG pipeline components
- Record a Loom video walking through the notebooks, the questions, and your improvements!

# Share

- Make a social media post about your RAG evaluation learnings and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

Here's a template to get your post started!

```
Just went end-to-end from synthetic data generation to RAG evaluation with Ragas!

Key insights:
1. Knowledge graph-based SDG creates diverse, realistic test scenarios
2. Ragas metrics reveal specific weaknesses in retrieval AND generation
3. Small changes (chunk size, reranking) can dramatically improve quality

Building better AI systems through better evaluation!

A huge shoutout to @AI Makerspace for making this possible.

#AI #RAG #Evaluation #Ragas #LangChain #LangGraph #BuildInPublic
```

<details>
<summary><h3>Advanced Build (Optional): Semantic Chunking Strategy</h3></summary>

> **Note**: Completing an Advanced Build earns full credit **in place of** doing the base assignment notebook questions/activities.

##### MINIMUM REQUIREMENTS:

1. Baseline `LangGraph RAG` Application using `NAIVE RETRIEVAL`
2. Baseline Evaluation using `RAGAS METRICS`
   - [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
   - [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html)
   - [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html)
   - [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html)
   - [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html)
3. Implement a `SEMANTIC CHUNKING STRATEGY`
4. Create a `LangGraph RAG` Application using `SEMANTIC CHUNKING` with `NAIVE RETRIEVAL`
5. Compare and contrast results

##### SEMANTIC CHUNKING REQUIREMENTS:

Chunk semantically similar (based on designed threshold) sentences, and then paragraphs, greedily, up to a maximum chunk size. Minimum chunk size is a single sentence.

Have fun!

#### Submitting the Advanced Build:

1. Complete all steps of the Main Assignment above
2. Create a notebook that meets or exceeds the MINIMUM REQUIREMENTS as well as the SEMANTIC CHUNKING REQUIREMENTS
3. Add, commit and push your completed notebook to your repository

When submitting, provide:
- Your Loom video link demonstrating the semantic chunking implementation
- The GitHub URL to your completed notebook with the Advanced Build

</details>

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
2. Create a branch to track your changes. Example: `git checkout -b m07-assignment`
3. **IMPORTANT:** Start Cursor from the `07_Synthetic_Data_and_Evaluation` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
4. Answer all Questions using the `##### Answer:` markdown cell below them in both notebooks
5. Complete all Activities in both notebooks
6. Add, commit and push your modified notebooks to your GitHub repository. _NOTE: Do not merge it into your main branch._

When submitting your homework, provide:
- Your Loom video link
- The GitHub URL to the `07_Synthetic_Data_and_Evaluation` folder on your assignment branch
