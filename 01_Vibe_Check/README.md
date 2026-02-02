<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

> **Note:** While the examples in this module use the OpenAI API, please follow the best practices outlined in the [SRHG AI Usage Guidelines](https://srhg.enterprise.slack.com/docs/T0HANKTEC/F0AB86J3A1L).

<h1 align="center" id="heading">Module 1: Introduction and Vibe Check</h1>

### [Quicklinks](../00_AE_Quicklinks/README.md)

| 📰 Module Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
|[✨ Intro & Vibe Check](../00_Docs/Modules/01_Intro_and_Vibe_Check.md) | Coming soon! | [Session 1 Slides](https://www.canva.com/design/DAG_pEBegaA/l-UEN3U_Kt6e7iaHn6YBTQ/edit?utm_content=DAG_pEBegaA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Module 1 Assignment: Vibe Check](https://forms.gle/LGYqGJT9eS5tWkyN7) | [Feedback 1/27](https://forms.gle/wPQKAwXHaqo2V5aW6) |

## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** – these will be questions that you will be expected to gather the answer to! These can appear as general questions, or questions meant to spark a discussion during Part 1 and Part 2!

- 🏗️ **Activities** – these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** – Take on a challenge! These builds require you to create something with minimal guidance outside of the documentation. Completing an Advanced Build earns full credit in place of doing the base assignment notebook questions/activities.

### Main Assignment

In the following assignment, you are required to take the app that you created for the Agent Engineering - SRHG challenge (from [this repository](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge)) and conduct what is known, colloquially, as a "vibe check" on the application.

You will be required to submit a link to your GitHub, as well as screenshots of the completed "vibe checks" through the provided Google Form!


#### A Note on Vibe Checking

>"Vibe checking" is an informal term for cursory unstructured and non-comprehensive evaluation of LLM-powered systems. The idea is to loosely evaluate our system to cover significant and crucial functions where failure would be immediately noticeable and severe.
>
>In essence, it's a first look to ensure your system isn't experiencing catastrophic failure.

---

#### 🏗️ Activity #1: General Vibe Checking Evals

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner.
    - Aspect Tested: Explanation/teaching ability, simplification of complex technical concepts
2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested: Reading comprehension
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested: Creative writing
4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested: Mathematical reasoning and problem solving
5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested: Tone/style transformation and rewriting

#### ❓Question #1:

Do the answers appear to be correct and useful?
##### ✅ Answer:
Overall the anwers appear to be correct and useful. However, I found that my output format is a little off. 

---

#### 🏗️ Activity #2: Personal Vibe Checking Evals (Your Assistant Can Answer)

Now test your assistant with personal questions it should be able to help with. Try prompts like:

- "Help me think through the pros and cons of [enter decision you're working on making]."
- "What are the pros and cons of [job A] versus [job B] as the next step in my career?"
- "Draft a polite follow-up [email, text message, chat message] to a [enter person details] who hasn't responded."
- "Help me plan a birthday surprise for [person]."
- "What can I cook with [enter ingredients] in fridge."

##### Your Prompts and Results:
1. Prompt: help me think through the pros and cons of the hub and spoke model of transit gateway
   - Result:
If your environment requires centralized control, ease of management, and scalable connectivity among many VPCs or networks, the Hub and Spoke model is a great fit.
However, if your application demands low latency, heavy east-west traffic between spokes, or very high throughput with minimal bottlenecks, you may want to consider alternatives like partial meshes or direct peering in addition to the hub.

2. Prompt: when using a hub and spoke model for transit gateway, why do you see an increase in latency?
   - Result:
When using the hub-and-spoke model with a Transit Gateway, you typically see an increase in latency because all traffic between spoke networks (VPCs) must route through the central transit gateway (the hub). This additional hop introduces extra processing and network traversal time compared to direct VPC-to-VPC peering or a more optimized routing path.

3. Prompt: how can i decrease latency when using a hub and spoke model?
   - Result:
To decrease latency in a hub-and-spoke model, focus on optimizing routes and network infrastructure between the hub and spokes, leveraging edge processing where possible, upgrading connection quality, fine-tuning network protocols, and possibly introducing additional hubs to reduce distances. Monitoring performance will help iteratively improve latency outcomes.

#### ❓Question #2:

Are the vibes of this assistant's answers aligned with your vibes? Why or why not?
##### ✅ Answer:
The vibes of this assistant's answers align with my vibes because it gives concise and direct answers.

---

#### 🏗️ Activity #3: Personal Vibe Checking Evals (Requires Additional Capabilities)

Now test your assistant with questions that would require capabilities beyond basic chat, such as access to external tools, APIs, or real-time data. Try prompts like:

- "What does my schedule look like tomorrow?"
- "What time should I leave for the airport?"

##### Your Prompts and Results:
1. Prompt: Take a look at my google calendar and tell me what my schedule is like tomorrow.
   - Result:
I don’t have access to your Google Calendar. However, if you share your schedule or provide details about any events you have tomorrow, I can help summarize it for you!

2. Prompt: Can you check the weather for tomorrow?
   - Result:
I’m unable to access real-time data, including current weather updates. You can check the weather for tomorrow using a reliable weather website or app like Weather.com, AccuWeather, or your local meteorological service. If you tell me your location, I can guide you on how to find the forecast!

#### ❓Question #3:

What are some limitations of your application?
##### ✅ Answer:

The application has several key limitations:

  1. No external tool integration: The assistant cannot access real-time data or external services like Google Calendar, email,
   or weather APIs, limiting its ability to help with time-sensitive or personalized queries.
  2. No persistent memory: The assistant doesn't retain information across sessions, so it can't build context about my
  preferences or past interactions over time.
  3. Output formatting issues: As noted in Activity #1, the response formatting isn't always optimal and could be improved for
  better readability.
  4. Limited to text-based interactions: The assistant cannot process images, files, or other media types that might be useful
  for certain tasks.
  5. No proactive capabilities: It can only respond to prompts rather than proactively alerting or reminding me about important
   information.
---

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

<details>
<summary><h3>🚧 Advanced Build (OPTIONAL)</h3></summary>

Please make adjustments to your application that you believe will improve the vibe check you completed above, then deploy the changes to your Vercel domain [(see these instructions from your Challenge project)](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge/blob/main/README.md) and redo the above vibe check.

> NOTE: You may reach for improving the model, changing the prompt, or any other method.

#### 🏗️ Activity #1
##### Adjustments Made:
- _describe adjustment(s) here_

##### Results:
1. _Comment here how the change(s) impacted the vibe check of your system_
2.
3.
4.
5.

</details>


## Submitting Your Homework
### Main Assignment
Follow these steps to prepare and submit your homework:
1. Pull the latest updates from upstream into the main branch of your Agent Engineering - SRHG repo:
    - For your initial repo setup see [Initial_Setup](https://github.com/AI-Maker-Space/Agent Engineering - SRHG/tree/main/00_Docs/Prerequisites/Initial_Setup)
    - To get the latest updates from AI Makerspace into your own Agent Engineering - SRHG repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `01_Prototyping_Best_Practices_and_Vibe_Check` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Edit this `README.md` file
4. Complete all three Activities:
    - **Activity #1:** Evaluate your system using the general vibe checking questions and define the "Aspect Tested" for each
    - **Activity #2:** Test your assistant with personal prompts it should be able to answer
    - **Activity #3:** Test your assistant with prompts requiring additional capabilities
5. Provide answers to all three Questions (`❓Question #1`, `❓Question #2`, `❓Question #3`)
6. Add, commit and push your modified `README.md` to your origin repository's main branch.

When submitting your homework, provide the GitHub URL to your Agent Engineering - SRHG repo.

### The Advanced Build:
1. Follow all of the steps (Steps 1 - 6) of the Main Assignment above
2. Document what you changed and the results you saw in the `Adjustments Made:` and `Results:` sections of the Advanced Build
3. Add, commit and push your additional modifications to this `README.md` file to your origin repository.

When submitting your homework, provide the following on the form:
+ The GitHub URL to your Agent Engineering - SRHG repo.
+ The public Vercel URL to your updated Challenge project on your Agent Engineering - SRHG repo.
