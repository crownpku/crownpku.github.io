---
layout: post
comments: true
title: Building My "Digital Twin":How I Distilled 27,000 Lines of Chat History into a Local LLM Wiki**
published: true
---

In the rapidly evolving landscape of Agentic AI, I decided to run a personal experiment. What happens if you take your entire conversation history with an AI assistant and ask a local LLM agent to distill *you* into a structured, compounding knowledge base? 

Armed with the **Gemini CLI** and an export of my data, I successfully "distilled" my professional and personal life into a localized AI brain. 

Here is a breakdown of how I did it, why it challenges traditional RAG (Retrieval-Augmented Generation), and what my digital twin actually learned about me.

### Phase 1: The Experiment and the Insights
To get the data, I used Google Takeout to export a massive 27,190 lines of JSON file of my Gemini chat history (yes I talk to Gemini probably more than I talk to real people LOL). I then fed this raw data into my local terminal using the Gemini CLI. 

Watching the AI parse through the data and automatically map out my life into an Obsidian vault was both awe-inspiring and highly entertaining. It didn't just summarize facts; it ruthlessly analyzed my workflows, tech grievances, and problem-solving habits. 

Here are a few highlights of what my digital clone synthesized:
*   **The Professional Summary:** It instantly consolidated my career timeline, highlighting my roles, and my past and latest personal projects in Agentic AI and model fine-tuning - all into a single file of `Guan_Wang.md`.
*   **The Ultimate CRM:** It accurately mapped out my family structure, creating a dedicated entity file for my wife, and noting details about our two children and our family travel preferences.
*   **The Tech Grievance Log:** Under a newly generated `Hardware_Preferences.md` file, the AI documented my loyalty to the Apple ecosystem, but also meticulously immortalized my petty struggles with a stubborn pair of AirPods 4 that kept flashing an amber error light.
*   **Strategic "Travel Hacking":** In `Travel_Optimization.md`, it saw right through my consumer behavior. It documented a highly tactical plan to buy a DJI Osmo Action 6 and a 2TB SSD in Hong Kong, explicitly noting that I timed the purchase to trigger a specific credit card cashback offer. 
*   **Aspirational Vlogging:** It captured my ambition to shoot rugged, minimalist videos following some of my favourite Youtubers, alongside the reality that I mostly just ride a gravel bike around Singapore.
*   **Financial & Health Logs:** It accurately synthesized my Dollar Cost Averaging (DCA) investment strategies and even organized my recent medical recovery logs into a structured profile.

### Phase 2: What Exactly is an LLM Wiki?
This project was heavily inspired by AI visionary Andrej Karpathy's open-source "LLM Wiki" concept. 

Most of us use **RAG (Retrieval-Augmented Generation)** to interact with documents. RAG works well, but it has a "goldfish memory" - the LLM is forced to rediscover and piece together knowledge from scratch on every single query. 

The **LLM Wiki framework** flips this paradigm. Instead of retrieving from raw documents at query time, the LLM incrementally builds and maintains a persistent, compounding wiki. The architecture relies on three layers:
1.  **Raw Sources:** The immutable data you feed it (in my case, the 27k-line Google Takeout export from my Gemini conversations and various PDFs).
2.  **The Wiki:** The LLM-maintained directory of Markdown files, including an `index.md`, a chronological `log.md`, and folders for Entities and Concepts.
3.  **The Schema:** The set of rules telling the LLM how to structure the wiki, format files, and maintain cross-references.

### Phase 3: The Result
Whenever I feed this local system new data, the Gemini CLI acts as a tireless librarian. It reads the source, extracts new facts, updates existing concept pages (like `Travel_Optimization.md` or `Investment_Strategy.md`), flags contradictions, and logs its own actions. 

This experiment completely changed how I view Personal Knowledge Management (PKM). We are moving past the era of using AI merely as an advanced search engine, and stepping into a world where AI acts as a dedicated, proactive research assistant that helps our knowledge actually *compound* over time.

Have you experimented with Agentic workflows or local LLM wikis yet? I'd love to hear how you're managing your personal or team knowledge bases!