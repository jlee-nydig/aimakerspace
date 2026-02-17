<p align="center" draggable="false"><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Module 5: Multi-Agent Applications</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE9/tree/main/00_AIE_Quicklinks)

| Module Sheet | Recording     | Slides        | Repo         | Homework      | Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Multi-Agent Applications](../00_Docs/Modules/05_Multi-Agent_Applications/README.md) |Coming soon! | [Module 5 Slides](https://www.canva.com/design/DAHA9EgZhrs/yMlmJQwBRCFHxvPvVyhlTQ/edit?utm_content=DAHA9EgZhrs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Module 5 Assignment: Multi Agents](https://forms.gle/nXFVqwvVfiVM8DWQ7) | [Feedback 2/10](https://forms.gle/k1cESPgXsYWrBTDU7) |


### Outline:

**BREAKOUT ROOM #1:**
- Task 1: Dependencies & Environment Setup
- Task 2: Understanding Multi-Agent Systems
- Task 3: Building a Supervisor Agent Pattern
- Task 4: Adding Tavily Search for Web Research
     - **Activity #1**: Add a Custom Specialist Agent

**BREAKOUT ROOM #2:**
- Task 5: Agent Handoffs Pattern
- Task 6: Building an Investment Agent Team
- Task 7: Context Engineering & Optimization
- Task 8: Visualizing and Debugging with LangSmith
     - **Activity #2**: Implement Hierarchical Teams

### Prerequisites:

#### 1. API Keys Required

You'll need API keys for:
- **OpenAI** - For GPT-5.2 (supervisor) and GPT-4o-mini (specialist agents)
- **Tavily** - For web search capabilities (free tier available at [tavily.com](https://www.tavily.com/))
- **LangSmith** (optional) - For tracing and debugging

### Steps to Run:

1. Install UV, which you can do through [this resource](https://docs.astral.sh/uv/#getting-started)
2. Run the command `uv sync`
3. Open your Jupyter notebook and select `.venv` for your kernel.

# Build

Run the notebook!

# Ship

- Customize your multi-agent system with one of the following enhancements:
     - Add a new specialist agent to the team (e.g., a Regulatory Agent, ESG Agent)
     - Implement a different multi-agent pattern (Network/Swarm, Debate)
     - Add human-in-the-loop approval for certain agent decisions
     - Implement context summarization to manage long conversations
- Create a diagram showing your multi-agent architecture
- Record a Loom video walking through the notebook, the questions, and your enhancements!

# Share

- Show your multi-agent architecture diagram in a Loom video and explain the agent interactions
- Make a social media post about your multi-agent investment advisory system and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

Here's a template to get your post started!

```
Built my first Multi-Agent AI System using LangGraph!

What I learned about multi-agent patterns:
1. Supervisor pattern = orchestrator that routes to specialist agents
2. Handoffs = agents can transfer control to each other based on expertise
3. Context engineering is critical - garbage in, garbage out

Key insight: Don't build multi-agents unless you really need them!
Start simple, add complexity only when necessary.

A huge shoutout to @AI Makerspace for making this possible.

#AI #Agents #LangGraph #MultiAgent #BuildInPublic
```

<details>
<summary><h3>Advanced Build (Optional): Investment Research Planner with File I/O</h3></summary>

> **Note**: Completing an Advanced Build earns full credit **in place of** doing the base assignment notebook questions/activities.

Build an **Investment Research Planner** - a multi-agent system that can create, save, and manage personalized investment research reports using file system tools.

### Requirements

**1. Multi-Agent Architecture:**
- A **Planner Supervisor** (GPT-5.2) that coordinates the research process
- Specialist agents for each investment domain (Market Outlook, Investment Strategy, Risk Management, Performance Analysis)
- A **File Manager Agent** that handles reading/writing investment reports

**2. File System Tools:**
Create tools that allow agents to:
- `save_investment_report(filename, content)` - Save a report to a markdown file
- `load_investment_report(filename)` - Load an existing report
- `list_saved_reports()` - List all saved investment reports
- `append_to_report(filename, section, content)` - Add a section to an existing report

**3. Workflow:**
```
User: "Create an investment analysis for a moderate-risk investor seeking long-term growth"
                           |
                           v
                 +-------------------+
                 | Planner Super-    |
                 | visor (GPT-5.2)   |
                 +---------+---------+
                           | coordinates
         +----------------+-+----------------+
         |                |                  |
         v                v                  v
   +----------+    +-----------+    +-------------+
   | Market   |    | Strategy  |    |    Risk     |
   | Outlook  |    |  Agent    |    | Management  |
   +----+-----+    +----+------+    +------+------+
        |               |                  |
        +---------------+------------------+
                        v
               +-------------------+
               |  File Manager     |
               |     Agent         |
               +---------+---------+
                         |
                         v
              investment_report_2025.md
```

**4. Example Output File (`reports/investment_analysis_moderate_growth.md`):**
```markdown
# Investment Analysis: Moderate-Risk Long-Term Growth
Generated: 2025-01-26

## Market Outlook
- Current economic environment: [analysis]
- Key macro trends to watch
- Market risks and opportunities

## Recommended Strategy
- Target allocation: 60% equities, 25% fixed income, 15% alternatives
- Sector overweights and underweights
- Geographic diversification approach

## Risk Management
- Tail risk hedging recommendations
- Rebalancing triggers and frequency
- Downside protection strategies

## Performance Benchmarks
- Target return: 7-9% annualized
- Risk metrics: Sharpe ratio > 0.8, max drawdown < 15%
- Comparison benchmarks: 60/40 portfolio, S&P 500

## Quarterly Review Template
- [ ] Portfolio rebalancing check
- [ ] Risk metric review
- [ ] Market outlook update
```

### Bonus Features (optional)
- Add a **Performance Tracker Agent** that can update reports with actual results
- Implement **report versioning** (save revisions with timestamps)
- Add **web search** to include latest market research in reports

### Resources
- [LangGraph Hierarchical Teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)
- [LangChain File System Tools](https://python.langchain.com/docs/integrations/tools/filesystem/)

### Submitting the Advanced Build
1. Complete all steps of the Main Assignment above
2. Include your multi-agent implementation with file I/O tools
3. Include at least 2 example generated investment reports in a `reports/` folder
4. Document your architecture with a diagram
5. Add, commit and push your modifications to your repository

When submitting, provide:
- Your Loom video link demonstrating the planner creating and saving an investment report
- The GitHub URL to your completed notebook with the Advanced Build
- Screenshots of generated report files

</details>

# Submitting Your Homework

## Main Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:
    - _(You should have completed this process already.)_ For your initial repo setup, see [Initial_Setup](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Docs/Prerequisites/Initial_Setup)
    - To get the latest updates from AI Makerspace into your own AIE9 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `05_Multi_Agent_with_LangGraph` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Answer Questions 1 - 4 using the `##### Answer:` markdown cell below them.
4. Complete Activity #1 and Activity #2 in the notebook.
5. Add, commit and push your modified `Multi_Agent_Applications_Assignment.ipynb` to your GitHub repository.

When submitting your homework, provide:
- Your Loom video link
- The GitHub URL to your completed notebook
