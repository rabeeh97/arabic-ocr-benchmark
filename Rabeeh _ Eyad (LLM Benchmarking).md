# Project Description

Arabic OCR in cursive is an interesting topic that is mostly academic for now. Let’s dip our toes there.  
~~I will set up a Shared Google Drive under [Mawsool.ai](http://Mawsool.ai) workspace account for both of us.~~

# Phase\#1 \- Benchmarking

The first thing to try is building a benchmark for the current state of the art. My hypothesis here is that LLM are taking over traditional methods, and the availability of OpenAI Fine-tuning and Gemini Vertex AI fine tuning could be promising if one has access to data.

Benchmarking is about verifying the performance of the chosen LLMs/technologies for the SAME set of data.

This means we have to consider:

- Datasets: which datasets we have available to benchmark the models  
- Models: which models to choose, how we could have access, and how we could simplify calling all of them with the same tools (not to build a solution for each model).  
- Performance: which metrics to consider: WER and CER, API speed, as well as Diacteric accuracy (for Arabic specifically), etc.

## **Datasets**

I found multiple datasets while searching

### Muharaf

Arxiv: [\[2406.09630\] Muharaf: Manuscripts of Handwritten Arabic Dataset for Cursive Text Recognition](https://arxiv.org/abs/2406.09630)  
Github: [GitHub \- MehreenMehreen/ScribeArabic](https://github.com/MehreenMehreen/ScribeArabic) (project of the data annotation)  
Github: [Muharaf dataset and related code](https://github.com/MehreenMehreen/muharaf) (data itself, and documentation on how to interpret this).  
Github: [GitHub \- MehreenMehreen/start\_follow\_read\_arabic](https://github.com/MehreenMehreen/start_follow_read_arabic) (all related, somehow).  
HuggingFace: [aamijar/muharaf-public · Datasets at Hugging Face](https://huggingface.co/datasets/aamijar/muharaf-public)  (Data is there and can be loaded directly using HuggingFace: meaning quick on Drive).  
Also Data: [https://zenodo.org/records/11492215](https://zenodo.org/records/11492215)   
I will provide the data if needed in other ways.

This is a Gemini Generated report: [OCR Research](https://docs.google.com/document/d/1KboDlTXd9aMAB8KlruNpVvTg-6E6TCKYxqCkhrs9VYI/edit?tab=t.0)in which the following section was created:

##### I will collapse this for readability

* **Muharaf:** This is a particularly valuable resource for historical Arabic manuscripts. It comprises over 1,600 fully annotated and transcribed page images, featuring a wide diversity of handwriting styles (predominantly Ruq'ah script, dating from the early 19th to the early 21st century) and various document types, including personal letters, diaries, notes, poems, church records, and legal correspondences.1 Annotations are detailed, providing spatial polygonal coordinates for text lines and marking basic page elements like graphics, page numbers, and headings. Muharaf is classified as a category 2 dataset, meaning it consists of original historical manuscripts rather than text scribed under controlled conditions.58  
* **IFN/ENIT:** A widely recognized benchmark dataset in the Arabic HTR community, often used for evaluating systems on handwritten Arabic, particularly for tasks like recognizing Tunisian town and village names.1  
* **AMFDS (Arabic Multi-Fonts Dataset):** This dataset is used for training and testing systems on a variety of Arabic fonts, including those designed to mimic handwritten styles, thus helping models generalize across different printed representations of Arabic text.17  
* **Historical Arabic Handwritten Text Recognition Dataset (Mendeley Data):** This dataset offers 40 meticulously transcribed historical Arabic pages, each selected from distinct books and representing different geographical origins and centuries, contributing to deep learning OCR modeling for historical Arabic texts.1  
* **KHATT (Kalamu Waraqa Handwritten Arabic TexT):** Another important benchmark dataset for Arabic script recognition, providing a collection of handwritten Arabic text images for training and evaluation.1 Surveys by Lorigo and Govindaraju 1 and AlKendi et al. 1 also provide overviews of earlier datasets and HTR efforts for Arabic.

 

## Models

I want to start with Gemini and OpenAI and LlaVa (open source from Meta), also (Nada2125/paligemma-3b-arabic-handwritten-ocr-1)  as it’s easy to get API keys for them. Then I want to follow it with your chosen models, a combination of open models, and arabic ones like AllaM, or whatever comes from Humain (Saudi) or whatever.

Let me know if you need api keys.

I recommend starting with only one, to build intuition and think of the metrics before adding more.  
In the **LLM API Tooling Suggestion** tab: [Hadeel / Eyad (LLM Benchmarking)](https://docs.google.com/document/d/177oQn1_QjjnIxFErMxsoy0Z8amCshvdXHUlWduq7fZA/edit?tab=t.nnokunk6y7p6), it’s suggesting some libraries where we could avoid repeating code. I’m OK with either choice. As long as our choice of LLMs is available there.

In the **LLM IDE Tooling Suggestion** tab [Hadeel / Eyad (LLM Benchmarking)](https://docs.google.com/document/d/177oQn1_QjjnIxFErMxsoy0Z8amCshvdXHUlWduq7fZA/edit?tab=t.bwdr2vwxd81g), there is a development flow that you might prefer. Use that to seed your thinking and choose whatever you feel like.

## Performance

For the metrics we could also leverage Python libraries, mentioned in the API tooling: 

**Leverage Libraries:** Use Python libraries to do the heavy lifting. The `jiwer` library is excellent for calculating WER and CER.

Let’s start with simple metrics and try to keep track of the latency.  
Add to that the cost per request.

## Breakdown:

1. Try to download the Muharaf dataset and ONLY use 10 pictures and their texts (download it in colab, unzip it there, and download it from there to your machine directly for reference).  
2. Try it against any LLM of your choice. Start with Gemini or OpenAI.  
   1. Upload the pictures and ask it to transcribe. Example: [Nada OCR Gemma Based.ipynb](https://colab.research.google.com/drive/1Qej5akK6Rkf1sVKlioIbyzlFdWIR1TET#scrollTo=x__tDAKDK6yE)  
   2. Find a way to read the dataset expected output.  
   3. Compare using JIWER  
   4. Try to find a way to track the latency.  
3. Output data to Google Drive for check-points.  
4. Build a Table.

If all goes well, we have a good setup already. Let’s keep it on this one dataset and now do it 10% of those images.

5. Estimate costs while we are at it. I don’t know how much it will cost to do it for all those images; especially on a billed API. Colab costs are going to be manageable. I could buy you credit.  
6. Do we want to change the prompting by sending more metadata (stuff in the json files)?  
7. 10% is important, because if we decided to fine-tune and run it again, we would want to fine-tune on 90% of the data, and keep the 10% separate and benchmarkable (benchmarking should NEVER mix with training data).  
8. In this section [https://github.com/MehreenMehreen/muharaf?tab=readme-ov-file\#the-dataset](https://github.com/MehreenMehreen/muharaf?tab=readme-ov-file#the-dataset) the author mentioned a restricted dataset. Maybe we could use that for benchmarking/training as well.

If all goes well, let’s add other datasets:

9. Keep the original muharaf raw data from previous steps; we need to properly sum up and average across all datasets (it’s not simply the average of the average of each datasets. Each dataset’s size will determine its weight in the total average.  
10. Find other datasets from the above collapsed section (from the OCR Research).

If all goes well

11. fine-tune using Open AI \- Fine Tuner or Vertex AI. Those two provide different approaches and it’s interesting to try them.

# Random Stuff

[AlgoMap.io \- Free Data Structures & Algorithms Roadmap](https://algomap.io/roadmap)

[https://colab.research.google.com/drive/1xHa0BrkBCvhaE0rfv1IxRF7VFA12y\_gV\#scrollTo=rd3oDMfdg0hJ](https://colab.research.google.com/drive/1xHa0BrkBCvhaE0rfv1IxRF7VFA12y_gV#scrollTo=rd3oDMfdg0hJ)

[https://gemini.google.com/app/074cc655a0f9902f](https://gemini.google.com/app/074cc655a0f9902f)

[https://gemini.google.com/app/3af5858dcba1d4fc](https://gemini.google.com/app/3af5858dcba1d4fc)

[https://darkcoding.net/software/personal-ai-evals-aug-2025/?utm\_source=tldrnewsletter](https://darkcoding.net/software/personal-ai-evals-aug-2025/?utm_source=tldrnewsletter)

A Blogpost is a very expensive search query to find interesting people and get them to send interesting things to your inbox (just the concept  i liked).

[https://www.henrikkarlsson.xyz/p/search-query?utm\_source=substack\&utm\_medium=email](https://www.henrikkarlsson.xyz/p/search-query?utm_source=substack&utm_medium=email)

Slavik Fine-Tuning of OCR.  
[https://x.com/wjb\_mattingly/status/1953263017296056365?utm\_source=substack\&utm\_medium=email](https://x.com/wjb_mattingly/status/1953263017296056365?utm_source=substack&utm_medium=email)

Ocr tweet:  
[https://x.com/wjb\_mattingly/status/1953263017296056365?utm\_source=substack\&utm\_medium=email](https://x.com/wjb_mattingly/status/1953263017296056365?utm_source=substack&utm_medium=email)

Arabic models and stuff:  
[https://arabic.ai/](https://arabic.ai/)  
[https://pronoia.tarjama.com/](https://pronoia.tarjama.com/)  
[https://www.middleeastainews.com/p/uae-tarjama-launches-ai-models](https://www.middleeastainews.com/p/uae-tarjama-launches-ai-models)  
[https://tarjama.com/introducing-pronoia-llm-the-arabic-ai-engine-built-for-enterprise-excellence/](https://tarjama.com/introducing-pronoia-llm-the-arabic-ai-engine-built-for-enterprise-excellence/)

Benchmarking  
[Ocrbench Leaderboard \- a Hugging Face Space by echo840](https://huggingface.co/spaces/echo840/ocrbench-leaderboard)   
What are the metrics for OCR.

LoRA training for images.  
https://github.com/ostris/ai-toolkit.git

Datasets:  
muharaf

# Learnings

## **Vision Course From Stanford CS231**

[Karpathy](https://docs.google.com/document/d/154XpG6u_LUEDQYGElI30RkRx4ljl07tKwNjQSvPJAok/edit?tab=t.0#heading=h.girq6f7vm905): my document where i wrote some notes about it. If you want only the courses, this is the link: [CS231n Winter 2016: Lecture1: Introduction and Historical Context](https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=1)

# Tokenization is super Important

[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=7840s&ab_channel=AndrejKarpathy) I loved this video. Check out Tokenization; this is a very important topic for your work.

# LLM API Tooling Suggestion

Of course. It sounds like you've hit a common but frustrating roadblock in applied AI work. The initial excitement of a project can quickly fade when you get bogged down by tedious engineering challenges. What you're experiencing is a classic "M-of-N" problem: benchmarking *M* datasets against *N* models, where the friction of each combination adds up quickly.

Let's break down the issues and explore some strategies to make this project more manageable and, hopefully, more fun again.

### **The Core Problems & Solutions**

You're facing three main challenges: varied APIs, inconsistent outputs, and slow iteration cycles (latency). Here’s a strategic approach to tackle each one.

---

### **1\. The "API Hell" Problem: Standardize Your Interactions**

Instead of writing custom code for every LLM, use a library that provides a single, unified interface. This is the single most important change you can make to your workflow.

**Top Recommendation: Use a Meta-API Library**

These libraries act as a middle-man, letting you call dozens of different LLMs using the exact same code format. You simply change a model identifier string to switch between them.

* **LiteLLM:** This is an excellent, lightweight, and open-source choice. It supports over 100 LLMs (including OpenAI, Anthropic, Cohere, and open-source models via Ollama or Replicate). You write your API call once and can loop through a list of models.  
* **LangChain / LlamaIndex:** These are more comprehensive frameworks for building LLM applications. They also provide standardized model interfaces and might be useful if you plan to add more complex logic later, but for pure benchmarking, LiteLLM is often simpler.

**How this helps:** Your code becomes cleaner and more scalable. Adding a new model to your benchmark suite is as simple as adding its name to a list, not writing a whole new integration.

### **2\. The Inconsistent Output Problem: Build a Robust Evaluation Pipeline**

To handle the messy and varied responses from different models, you need to treat this as a small software engineering project. A structured pipeline will save you immense time in the long run.

**A. Structure Your Project:**

/cursive-arabic-benchmark  
|-- data/  
|   |-- dataset1/  
|   |   |-- image001.png  
|   |   |-- image002.png  
|   |-- labels.csv  (image\_path, ground\_truth\_text)  
|-- results/  
|   |-- benchmark\_outputs.jsonl  
|-- main.py  
|-- analysis.ipynb

**B. Create a Repeatable Script (`main.py`):**

This script will be the heart of your operation.

1. **Load Data:** Read your `labels.csv` into a structured format.  
2. **Define Models:** Create a list of the model identifiers you want to test (e.g., `['gemini/gemini-1.5-pro-latest', 'openai/gpt-4o', 'anthropic/claude-3-opus-20240229']`).  
3. **Loop & Process:** Iterate through each image and each model.  
   * **Prompt Consistently:** Use a clear, simple prompt. For image-to-text, it might just be the image itself with a simple instruction like "Transcribe the Arabic text in this image."  
   * **Parse the Output:** This is critical. After getting a response, immediately clean it. Use regular expressions or string methods to remove conversational fluff (e.g., "Certainly, here is the transcription:") and extract only the Arabic script.  
   * **Store Everything:** For each run, save the image ID, the ground-truth text, the model used, the raw output, and the cleaned output to a single file (JSON Lines or CSV is great for this).

**C. Analyze the Results (`analysis.ipynb`):**

Once your script has run, you can analyze the `benchmark_outputs.jsonl` file.

* **Use Standard Metrics:** For transcription tasks, the industry standards are:  
  * **Character Error Rate (CER):** The percentage of characters that were incorrectly transcribed. This is often the most important metric for OCR.  
  * **Word Error Rate (WER):** Similar to CER but at the word level.  
* **Leverage Libraries:** Use Python libraries to do the heavy lifting. The `jiwer` library is excellent for calculating WER and CER.

Python  
\# Example using jiwer  
import jiwer

ground\_truth \= "النص الأصلي"  
model\_output \= "النص الأصلي"

\# Simple case  
transformation \= jiwer.Compose(\[  
    jiwer.RemovemultipleSpaces(),  
    jiwer.Strip(),  
\])

cer \= jiwer.cer(ground\_truth, model\_output, truth\_transform=transformation, hypothesis\_transform=transformation)  
print(f"Character Error Rate: {cer:.2%}")

### **3\. The Latency Problem: Speed Up Your Iterations**

Waiting for APIs is unavoidable, but you don't have to do it one by one.

* **Embrace Asynchronous Requests:** This is a game-changer. Instead of sending one request and waiting for the response before sending the next, you can send many requests in parallel. Python's `asyncio` library is designed for this. Libraries like **LiteLLM and LangChain have built-in support for asynchronous calls**, which makes this much easier to implement. This will dramatically cut down your total runtime.  
* **Start Small:** Debug your entire pipeline on a tiny subset of your data (e.g., 5-10 images). Once you're confident everything works, you can launch the full benchmark and let it run.  
* **Use Caching:** If you find yourself re-running the same image against the same model, implement a simple caching mechanism to save the result and avoid redundant API calls.

### **Summary: Your New Workflow**

1. **Pick a Tool:** Choose **LiteLLM** to handle all your API calls.  
2. **Structure:** Organize your project with distinct data, results, and code folders.  
3. **Build the Engine:** Write a Python script (`main.py`) that uses **asynchronous calls** to loop through your images and models, parsing and saving the results systematically.  
4. **Analyze:** Use a Jupyter Notebook (`analysis.ipynb`) with libraries like **pandas** and **jiwer** to load your results and compute the CER/WER for each model, creating tables and charts to compare performance.

By investing a little time in building this structured workflow, you'll automate the tedious parts and free yourself up to focus on the interesting analysis. This will make your project more scalable, your results more reliable, and the entire process far more satisfying.

# LLM IDE Tooling Suggestion

For this project, **use VSCode as your primary development environment.**

It offers the best combination of features for what you need to do: developing a script, managing files, and analyzing the results, all in one place.

Here’s a breakdown of why it's the best fit and the roles of the other tools you mentioned.

---

### **\#\# Why VSCode is Your Best Choice 🏆**

VSCode is a full-fledged Integrated Development Environment (IDE) that gives you the power and flexibility needed for a project with multiple files and scripts.

* **Project & File Management:** Your project involves separate folders for `data` and `results`, a main Python script (`main.py`), and an analysis notebook (`analysis.ipynb`). VSCode's file explorer makes managing this structure seamless.  
* **Integrated Terminal:** You can open a terminal directly inside VSCode to install libraries (`pip install liteLLM jiwer`) and run your main benchmarking script (`python main.py`).  
* **Excellent Notebook Support:** VSCode has a world-class Jupyter Notebook editor built-in. You can write and run your `analysis.ipynb` directly within the same window as your other project files, creating a smooth workflow from running the benchmark to visualizing its results.  
* **Debugging Tools:** When your script inevitably has a bug, VSCode's debugger is far more powerful than `print()` statements, allowing you to step through your code line-by-line to see what's going wrong.  
* **Extensibility:** You can integrate AI coding assistants like **GitHub Copilot** directly into VSCode. This is like having Claude or another LLM assistant embedded right in your editor, providing real-time suggestions and boilerplate code.

---

### **\#\# How the Other Tools Fit In**

While VSCode should be your main workbench, the other tools can serve specific, complementary purposes.

#### **Google Colab 🧪**

* **Good for:** Analysis and experimentation, especially if you don't want to set up a Python environment locally.  
* **How to use it:** You could do your final analysis (`analysis.ipynb`) in Colab. After running your main script locally and generating the `results.jsonl` file, you can upload it to Google Drive and open it in a Colab notebook for visualization and metric calculation.  
* **Limitation:** It's clunky for developing a multi-file project and managing datasets. It's best thought of as a cloud-based notebook, not a full IDE.

#### **Claude Code (CLI) / AI Assistants 🤖**

* **Good for:** Code generation and debugging assistance.  
* **How to use it:** Think of this not as an environment, but as a consultant. When you're stuck on how to write your asynchronous API calls or a tricky parsing function, you can ask an AI assistant to generate the code snippet for you. You would then paste this code into your script in **VSCode**. The best way to do this is with an integrated tool like GitHub Copilot.

#### **NotebookLM 🧠**

* **Good for:** Research and understanding, **not for coding**.  
* **How to use it:** Before you start, you could upload the API documentation for the LLMs you're testing into NotebookLM. You could then ask it questions like, "What are the rate limits for the Gemini 1.5 Pro API?" or "Summarize the key differences between these three APIs." It helps you process your source material.  
* **Limitation:** **NotebookLM does not execute code.** You cannot run your benchmark in it.

---

### **\#\# Recommended Workflow**

1. **Setup in VSCode:** Install VSCode and the Python and Jupyter extensions. Create your project folder and file structure.  
2. **Code in VSCode:** Write your `main.py` script. Use the integrated terminal to install packages.  
3. **Get Help from AI:** Use an extension like GitHub Copilot or a separate tool like Claude to help you write complex parts of your code.  
4. **Run in VSCode:** Execute your `main.py` script from the integrated terminal to run the full benchmark and generate your `results.jsonl` file.  
5. **Analyze in VSCode:** Open your `analysis.ipynb` notebook in VSCode. Load the results file and perform your analysis with libraries like Pandas and Jiwer to create tables and charts.

# Next step

I was searching for an alternatives of Jiwer, and i want to try those:

* CAMeL-Lab:  
  [https://github.com/CAMeL-Lab/camel\_morph](https://github.com/CAMeL-Lab/camel_morph)  
  morphological models for Arabic  
    
* Custom Normalizers \+ jiwer:  
  stick with jiwer but make it Arabic-friendly  
    
* SCTK:  
  [https://github.com/usnistgov/SCTK](https://github.com/usnistgov/SCTK)  
  More accurate than Jiwer but harder to install

Need API\_KEY for gemini pro the free trial over after some tries  
   
Try new dataset less cursive than Muharaf

Should I keep track of prompt id or latency ?

—---------------------------

* **Do we need Arabic normalization, is it a good choice for benchmarking ?**

* **Do we need the marco wer and cer ot is it enough for global ?**  
  **Global** answers: “How good is the OCR overall?”  
  **Macro** answers: “How consistently good is it across samples?”

# Cost

* **image pricing in Gemini** is handled differently from how you might expect if you’ve used OpenAI’s models before.  
* gemini-2.5-flash: does not tokenize images pixel into tokens, instead it uses a **multimodal encoder** that embeds the image into something like dense vector not a sequence of tokens