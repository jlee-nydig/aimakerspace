# Module 8: 📊 Agentic RAG Evaluation

🎯 Learn to set up and implement effective evals for agents and RAG applications.

- Build an Agentic RAG application with LangGraph
- Learn to evaluate RAG and Agent applications quantitatively with the RAG ASsessment (RAGAS) framework
- Use metrics-driven development to improve agentic applications, measurably, with RAGAS 

🎯 Learn to set up and implement effective evals for agents and RAG applications.

📚 **Learning Outcomes**
- Build an Agentic RAG application with LangGraph
- Learn to evaluate RAG and Agent applications quantitatively with the RAG ASsessment (RAGAS) framework
- Use metrics-driven development to improve agentic applications, measurably, with RAGAS 

🧰 **New Tools**
- Reranking: [Cohere Rerank](https://cohere.com/rerank)

## 📛 Required Tooling & Account Setup
- [Cohere API Key](https://dashboard.cohere.com/welcome/register)
- [Metals.dev](https://metals.dev/)

   
## 📜 Recommended Reading

- [LLM-as-a-Judge](https://en.wikipedia.org/wiki/LLM-as-a-Judge)
- [Self-Refine](https://arxiv.org/abs/2303.17651) (Mar 2023)
- [RAGAS](https://arxiv.org/abs/2309.15217) (Sep 2023)
- [Lessons from Improving AI Applications](https://blog.ragas.io/hard-earned-lessons-from-2-years-of-improving-ai-applications) (May 2025)
- [In Defense of Evals](https://www.sh-reya.com/blog/in-defense-ai-evals/) (Sep 2025)

# 🗺️ Overview

From the outset, as we investigate RAG evaluation and assessment, it’s important to note that the driving question behind LLM applications since ChatGPT came onto the market is “Are my LLM app outputs… good?  right?  correct?  useful?  unbiased?  reliable?”  The way we answer this question depends on the application we’re building and what task it completes or what problem it solves.

<aside>
🤔

What if we had a generic framework to answer the “Is my app good, right, or correct?” question?

</aside>

In module 8, we explore a generic framework that does exactly this, built specifically to assess Retrieval Augmented Generation (RAG) systems.  Meet RAG ASsessment (RAGAS), the leading-edge evaluation tool for RAG applications.  RAGAS ([YC24](https://www.ycombinator.com/companies/ragas)) is attempting to develop the industry standard for analyzing RAG applications, and they’ve made great strides and progress in the last year!

Today we’ll dig into the metrics we can use across RAG applications. We’ll look at their definitions and how they’re calculated, and reflect on how we would think about leveraging them!

## On Assessment in 2026

In 2026, RAG (Retrieval-Augmented Generation) assessment is a mature technology, with evaluation frameworks like RAGAS at the forefront. As organizations increasingly integrate RAG into production environments, the demand for robust and scalable assessment tools has grown.

With more complex LLM (Large Language Model) applications, evaluating them requires deeper analysis. The transition from a "Wild West" approach in early years to a more structured, metrics-driven methodology reflects the industry's shift toward ensuring reliable and cost-effective RAG solutions.

New trends in evaluation include:

- **Test-time compute considerations** – balancing performance with cost.
- **Multi-turn and agent evaluation** – expanding beyond single-turn RAG applications.
- **More robust benchmarks** – emphasizing both human and computational evaluation.
- **Integration with reasoning models** – improving assessment by allowing models to "think" through their evaluations.

As complexity increases, AI engineers must provide simpler, more interpretable evaluation methodologies while managing deeper technical intricacies.

## 📊 Metrics-Driven Development

Metrics-Driven Development (MDD) is a data-centric way to drive product development.  It requires evaluating and monitoring key metrics over time, ***especially before and after we change anything*** within our LLM application.

In short, we can break it down into three steps:

1.  Establish baseline
2. Change stuff that potentially improves retrieval
3. Recalculate metrics

Upon recalculation, we can ask ourselves whether or not improvement has occurred!

Note that MDD doesn’t require the absolute value of our metrics to be deeply meaningful; rather, the ***change*** in metrics matters.  Are we increasing or decreasing our metrics?

We can use this approach to assess the impact of changing components throughout our RAG application, from document loading, text splitting, and advanced prompting and retrieval strategies, to swapping out vector stores, embedding models, or LLM chat models.  We can even use these metrics to assess the impact of fine-tuning, which we’ll do later in the course!   

<p align="center">
  <img src="./images/ragas app.png" width="50%" />
</p>

A RAG application depicting how to use RAGAS metrics to drive product development.

## 🧵 RAGAS

In the last module, we learned how to create synthetic test data for evaluation. In this module, we leverage that data to calculate relevant metrics that will help us assess and improve the performance of our application.

### Types of Metrics

First, we should understand that there are [different types of metrics](https://docs.ragas.io/en/stable/concepts/metrics/overview/#why-metrics-matter). 

1. One distinction is end-to-end vs. component-level metrics
2. Another is LLM-based vs. non-LLM-based

- End-to-End metrics focus on inputs and outcomes; e.g., "Is the final answer correct?" "Did the agent achieve the goal?" and so on.
- Component-level metrics focused on specific aspects of the system. In a RAG application, we might focus on the performance of only the retriever, for instance.

Another way to frame these ideas is through the lens of outcome supervision vs. process supervision. In systematic processes with verifiably correct answers (think math problems), we can reward either steps of the process (the thinking/reasoning steps) or strictly assess final answers. This has been [well-studied by OpenAI](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/), [Google](https://arxiv.org/abs/2406.06592), [DeepSeek](https://arxiv.org/abs/2402.03300), and others.

On the other hand, LLM-based metrics use an LLM to "judge" something. Instead of relying on completely deterministic evaluators or on human annotators, we bring in the LLM to help assess.

While there are many non-LLM-based evaluation approaches you should always keep in your pocket, including simple metrics like "exact match" (e.g., character-by-character) and fuzzy/semantic matching (classic NLP vs. LLM-specific). We will focus on LLM-based metrics for our evaluations.

Finally, it's worth noticing some best practices laid out by the RAGAS team:

1. `Prioritize End-to-End Metrics`
2. `Ensure Interpretability`
3. `Emphasize Objective Over Subjective Metrics`
4. `Few Strong Signals over Many Weak Signals`

Check out their [overview of metrics](https://docs.ragas.io/en/stable/concepts/metrics/overview/) to go deeper.

### RAG Metrics

Assuming we've used the Knowledge Graph approach to SDG to generate [Question, Reference Context, Reference Response] triples, we’re ready to calculate our RAGAS metrics.

* Note that in addition to the Reference Context and Reference Response, we will also have the Retrieved Context and the Response generated by our RAG application.

We can break any RAG system into two parts: the `retriever` and the `generator`. As you've heard from us before, "as goes retrieval, so goes generation."

`Retriever` Metrics:
- Context Recall
- Context Entities Recall
- Noise Sensitivity
- Context Precision

To understand retriever metrics, remember that we want the right context in the context window at the end of the day.
- **Recall** asks "how many relevant items were retrieved?"
- **Precision** asks "how many retrieved items were (precisely) relevant?"
- **Noise** asks "how many retrieved items were actually NOT relevant?" 

`Generator` Metrics:
- Faithfulness
- Response Relevancy

To understand generator metrics, remember that RAG is all about avoiding hallucinations.
- Faithfulness asks, "Did you hallucinate, or were you faithful to the context?"
- Response relevancy asks, "Does this answer appropriately address the original question?"

The details of each computation can be viewed in detail by clicking the links above. Alternatively, you can check out the RAGAS docs [here](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/).

RAGAS was built because the industry needed an answer to the question “How do I know if my application is good?”  While it might not be THE answer, it is AN answer. 

Always remember that although some absolute values can be interpreted, this systematic evals process is not about absolute values out of the box (although over time you can develop upper and lower bounds for your system), but more about using the relative changes in metrics to drive development.

Additionally, if metrics are too accurate (99-100%), be suspicious. Look for other dimensions along which to optimize (cost, latency, performance).

### Agent Metrics

Agentic systems are harder to evaluate, not only because they are much more dynamic (e.g., agents vs. workflows), but also because the data we should be collecting or synthetically generating is much less clear than [question, context, response].

For this reason, the metrics we do have mimic those for RAG and remind us of classic binary ML techniques. We can work end-to-end by assessing whether an agent stayed on topic towards achieving its goal, or we can assess one component of an agent's behavior: tool calls.

Here are the metrics we will look at:

- [Topic adherence](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#topic-adherence)
- [Tool call accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#tool-call-accuracy)
- [Tool call F1](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#legacy-api-deprecated_1)
- [Agent goal accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#legacy-api-deprecated_1)

You'll notice that topic adherence uses a precision-and-recall approach, as does Tool call F1.

Thus, we've reached the limits of our ability to assess agents, [as the RAGAS team](https://vibrantlabs.com/blog/hello-world).

> We set out two years ago to improve AI systems. Inserting evaluations into the workflow to catch errors in development and not in production. We were accepted into YCombinator on that premise and received a $2.5M pre-seed round to build whatever comes next. 

...

> The traditional way of evaluating and training AI agents has hit a wall. Simulations are the future.

Their first release towards this agentic evaluation future is [PA Bench: Evaluating Web Agents on Real World Personal Assistant Workflows](https://vibrantlabs.com/blog/pa-bench)
