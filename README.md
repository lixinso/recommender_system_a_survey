# Recommender System - A Survey

Xinsong Li, Dec 2020 (Updated Feb 2026)

# Abstract

This survey provides a comprehensive summary of recommender system knowledge, spanning from traditional approaches to the latest advances driven by Large Language Models (LLMs) and Agentic AI.

The survey is organized into two main sections. **Section 1** covers traditional recommender systems, including Content-Based Filtering, Collaborative Filtering, hybrid approaches, deep learning methods, and established evaluation metrics. **Section 2** explores the paradigm shift occurring in 2024–2026, where generative and agentic architectures are replacing classical retrieval-and-ranking pipelines, introducing capabilities such as autonomous goal-driven reasoning, semantic generative retrieval, retrieval-augmented generation (RAG), and explainable AI through natural language and knowledge graphs.

---

# Section 1: Traditional Recommender Systems

## 1.1 Overview

Recommender Systems are usually classified into 2 types by:

- Content Based Filtering
- Collaborative Filtering

Depending on whether the model is learned from underlying data, there are 2 types:

- Model Based
- Memory Based

## 1.2 Use Cases

- **Movie Recommendation** — e.g., Netflix
- **Music Recommendation** — e.g., Last.fm, Pandora Radio
- **Product Recommendation** — e.g., Amazon
- **News Recommendation** — e.g., Google News, Toutiao
- **People Recommendation** — e.g., LinkedIn

## 1.3 System Architecture

![Netflix System Architecture](https://miro.medium.com/max/1400/1*qqTSkHNOzukJ5r-b54-wJQ.png)

Source: [Netflix Technology Blog — System Architectures for Personalization and Recommendation](https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8)

## 1.4 Algorithms

### 1.4.1 Cold Start Problem

The cold start problem occurs when the system lacks sufficient data about new users or new items to make accurate recommendations.

### 1.4.2 Collaborative Filtering

Collaborative Filtering is best suited to problems with known data on users, but lack of data for items or lack of feature extraction for items of interest. [[6]](https://towardsdatascience.com/recommendation-systems-a-review-d4592b6caf4b)

Collaborative Filtering approaches build a model from a user's past behavior (items previously purchased or selected/rated) as well as similar decisions made by others. [[10]](https://en.wikipedia.org/wiki/Recommender_system#Mobile_recommender_systems)

### 1.4.3 Content-Based Filtering

Content-Based Filtering recommends items similar to those a user has liked in the past, based on item features and attributes.

### 1.4.4 Knowledge-Based System

Knowledge-based recommender systems use explicit knowledge about items, user preferences, and recommendation criteria to generate suggestions.

### 1.4.5 Hybrid Approaches

Hybrid approaches combine multiple recommendation strategies (e.g., collaborative filtering and content-based filtering) to overcome the limitations of individual methods.

### 1.4.6 Interaction-Based Approaches

- **Item-Item**: Measures similarity between items based on user interaction patterns.
- **User-Item**: Models the relationship between users and items directly.
- **User-User**: Finds users with similar preferences and recommends items liked by similar users.

### 1.4.7 Other Techniques

- **LDA (Latent Dirichlet Allocation)**: Topic modeling applied to user preferences.
- **Ranking**: Learning-to-rank algorithms for ordering recommendations.
- **Tag-Based**: Leveraging user-generated tags for content categorization.
- **Category-Based**: Using item category hierarchies for recommendation.
- **Trending**: Incorporating temporal popularity signals.
- **Feedback System**: Utilizing explicit and implicit user feedback.

### 1.4.8 Hybrid Approach — Parallelized Hybridization Design

- Output of several existing implementations combined
- Least invasive design
- Some weighting or voting schema (weights can be learned dynamically; extreme case is switching)

![Parallelized Hybridization Design](res/parallelized_hybridization_design_idea.PNG)

**Weighted Hybridization:**

- How to derive weights?
  - Estimate, e.g., by empirical bootstrapping (historical data needed; compute different weightings; decide which does best)
  - Dynamic adjustment of weights (start with uniform weight distribution; for each user, adapt weights to minimize error of prediction)
- Minimize MAE

![Weighted MAE](res/parallelized_hybridization_design_weighted_MAE.PNG)

### 1.4.9 Deep Learning Approaches

#### Neural Collaborative Filtering (NCF)

[[30]](https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e) [[31]](https://arxiv.org/abs/1708.05031) He et al. proposed a Neural Collaborative Filtering algorithm by replacing the inner product with a neural architecture that can learn an arbitrary function from data. NCF is generic and can express and generalize matrix factorization under its framework. To supercharge NCF modeling with non-linearities, they propose to leverage a multi-layer perceptron to learn the user-item interaction function.

#### ALS — Alternating Least Squares

[ALS Implementation Example](https://github.com/microsoft/recommenders/blob/master/examples/00_quick_start/als_movielens.ipynb)

#### xTreme Deep Factorization Machines (xDeepFM)

xDeepFM combines explicit and implicit feature interactions for click-through rate prediction.

### 1.4.10 Reinforcement Learning for Recommender Systems

Reinforcement learning approaches model the recommendation problem as a sequential decision-making process, optimizing for long-term user engagement rather than immediate click-through rates.

## 1.5 Evaluation Metrics

### Rating Metrics

- **Root Mean Square Error (RMSE)** — Measure of average error in predicted ratings
- **R Squared (R²)** — How much of the total variation is explained by the model
- **Mean Absolute Error (MAE)** — Average absolute difference between predicted and actual ratings
- **Explained Variance** — How much of the variance in the data is explained by the model

### Ranking Metrics

- **Precision** — The proportion of recommended items that are relevant
- **Recall** — The proportion of relevant items that are recommended
- **Normalized Discounted Cumulative Gain (NDCG)** — Evaluates how well the predicted items for a user are ranked based on relevance
- **Mean Average Precision (MAP)** — Average precision for each user normalized over all users

### Classification Metrics

- **Area Under Curve (AUC)** — Integral area under the receiver operating characteristic curve
- **Logistic Loss (Logloss)** — The negative log-likelihood of the true labels given the prediction of a classifier

## 1.6 Model Selection and Optimization

Model selection involves choosing the best algorithm and hyperparameters for a given recommendation task, often through cross-validation and grid search.

## 1.7 Sparsity

The number of items sold on majority e-commerce sites is extremely large. The most active users will only have rated a small subset of the overall database. Thus, even the most popular items have very few ratings. [[10]](https://en.wikipedia.org/wiki/Recommender_system#Mobile_recommender_systems)

## 1.8 Churn Prevention

Recommender systems play a role in churn prevention by maintaining user engagement through personalized and timely recommendations.

## 1.9 Industry Practice

- **Last.fm** — Music recommendation based on listening history and social connections.
- **Pandora Radio** — Content-based music recommendation using the Music Genome Project.
- **LinkedIn** — [People You May Know (PYMK)](https://engineering.linkedin.com/teams/data/artificial-intelligence/people-you-may-know)

---

# Section 2: Latest Recommender System Trends with LLM/AI (2024–2026)

## 2.1 The Paradigm Shift: From Retrieval-Ranking to Generative-Agentic Architectures

The landscape of recommendation systems has undergone a fundamental transformation between 2024 and 2026, marking a definitive departure from the traditional retrieval-and-ranking pipelines that dominated the previous decade. This evolution is characterized by a shift from passive, predictive models that optimize for short-term engagement metrics, such as click-through rates (CTR), toward generative and agentic architectures designed to understand and facilitate complex user objectives. While earlier systems operated as "black boxes" that mapped user and item identifiers into dense latent spaces to predict the "next click," modern AI in the recommendation domain now attempts to comprehend "what the user is trying to achieve."

This transition is driven by four primary technological trends:

1. The rise of **agentic recommenders**
2. The adoption of **generative retrieval** via semantic identifiers
3. The integration of **retrieval-augmented generation (RAG)** for temporal grounding
4. The emergence of **explainable AI** through natural language reasoning and knowledge graphs

The historical bottleneck of recommendation systems — the inherent separation between the candidate generation (retrieval) stage and the heavy-ranking stage — is being dismantled. Traditional architectures relied on two-tower models and approximate nearest neighbor (ANN) search, which were fundamentally limited by their inability to handle the "cold-start" problem and their lack of differentiability across the entire pipeline. In contrast, the current era leverages the autoregressive capabilities of Large Language Models (LLMs) to directly generate item identifiers, effectively collapsing the retrieval and ranking steps into a unified, end-to-end differentiable process.

## 2.2 Agentic Recommendation Systems

### 2.2.1 The Architectural Leap

The most significant shift in the 2024–2026 period is the transition from "Generative AI" — characterized as creative but passive and "read-only" — to "Agentic AI," which is functional, active, and "read-write." Traditional recommendation engines were designed to wait for a user to scroll or provide an explicit query before presenting a static list of items. Agentic systems, however, are characterized by autonomy, adaptability, and goal-driven reasoning.

### 2.2.2 Proactive Autonomy and Multi-Agent Coordination

In an agentic recommendation framework, the system acts as an autonomous agent that can decompose a high-level goal into a series of actionable steps. For example, a travel recommendation request no longer simply returns a list of hotels based on historical preference. An agentic system reasons through the request by checking real-time flight availability, cross-referencing the user's personal calendar, analyzing recent weather reports at the destination, and synthesizing these data points to suggest a comprehensive travel package.

Multi-agent systems (MAS) further enhance this by introducing specialized roles within the recommendation pipeline. Architectures are emerging where a "budget agent" may focus on cost optimization, a "quality agent" prioritizes user reviews and brand reputation, and a "manager agent" synthesizes these competing priorities into a balanced recommendation.

**Comparison: Traditional vs. Agentic Recommender Systems**

| System Property | Traditional Recommendation (Pre-2023) | Agentic Recommender (2025–2026) |
|---|---|---|
| Operational Mode | Passive; reacts to user actions | Proactive; autonomous goal pursuit |
| Cognitive Model | Statistical pattern matching | Reasoning, planning, and self-correction |
| Interaction Style | Static list presentation | Conversational, plan-oriented dialogue |
| Tool Utilization | Internal database queries only | Dynamic use of external APIs and sensors |
| Architectural Basis | Monolithic models; Retrieval → Ranking | Multi-agent frameworks; Actor systems |

### 2.2.3 Multi-Agent Architecture Search (MaAS) and the Agentic Supernet

The design of complex multi-agent systems has evolved from labor-intensive manual engineering toward automated frameworks such as Multi-agent Architecture Search (MaAS). Rather than seeking a static, one-size-fits-all architecture, MaAS optimizes an "Agentic Supernet" — a probabilistic and continuous distribution of agentic architectures. During inference, the system can dynamically sample a query-dependent agentic sub-network, effectively allocating computational resources based on the difficulty of the task.

Experimental evaluations of the MaAS framework indicate that this automated, adaptive approach requires only **6% to 45%** of the inference costs of traditional handcrafted multi-agent systems while simultaneously improving performance quality by up to **11.8%**. This shift toward "Adaptive Inference" allows enterprises to deploy sophisticated agentic recommenders at scale without prohibitive costs.

## 2.3 Generative Retrieval and the Semantic Indexing Paradigm

### 2.3.1 From "Select from Database" to "End-to-End Generation"

A fundamental technical evolution is the move toward Generative Retrieval, which replaces the conventional "select from database" mechanism with "end-to-end generation." In traditional systems, items were represented by random atomic identifiers (e.g., Item ID #4092), which carried no inherent meaning. Modern systems, such as the Transformer Index for Generative Recommenders (TIGER), utilize "Semantic IDs" — discrete tokens derived from an item's content features, such as text descriptions, images, and metadata.

### 2.3.2 Hierarchical Quantization and Semantic ID Creation

The creation of Semantic IDs involves a two-stage process:

1. An item's content is mapped into a dense embedding using a pre-trained encoder (e.g., SentenceT5).
2. This dense representation is compressed into a sequence of discrete tokens using a Residual Quantized Variational AutoEncoder (RQ-VAE).

The quantization process is hierarchical, meaning each successive token in the ID provides a more granular refinement of the item's position in the semantic space.

Formally, given an item's dense embedding *z*, the RQ-VAE maps it to a tuple of codewords *(c₀, c₁, …, c_{m-1})* where each *c_d* is chosen from a codebook to minimize the residual error from the previous level. The probability of the model generating a specific item ID for the next user interaction is:

```
P(ID) = ∏_{k=0}^{m-1} P(c_k | c_0, …, c_{k-1}, history)
```

Because similar items share overlapping semantic tokens (prefixes), the model can generalize to new items with far greater efficiency than traditional systems.

**Generative Retrieval Frameworks Comparison**

| Framework | Mechanism | Impact on Cold Start | Key Technological Component |
|---|---|---|---|
| TIGER | Direct autoregressive Semantic ID decoding | Significant improvement; uses content features | RQ-VAE + Transformer Decoder |
| LIGER | Hybrid Generative + Sequential Dense Retrieval | Enhanced robustness; balances precision/recall | Dense Retrieval Reranker |
| RPG | Parallel long-sequence Semantic ID generation | Improved scalability; 12.6% NDCG@10 gain | Optimized Product Quantization (OPQ) |
| GENIUS | Multimodal Generative Retrieval | Broad generalization across text/image tasks | Modality-aware ID prefixing |

### 2.3.3 Mitigating the Cold-Start Problem through Semantic IDs

Generative retrieval addresses the cold-start problem by ensuring that the identifier itself is a distillation of the item's attributes. When a new item is introduced, its Semantic ID is generated immediately based on its title, brand, and description. The transformer model, having learned the relationships between semantic tokens during training, can correctly place the new item in the user's trajectory even without a single click.

Studies from Amazon Science and Spotify highlight that this approach not only improves retrieval of cold-start items but also enhances the overall diversity of recommendations by reducing the system's reliance on popularity bias. In some benchmarks, generative retrieval models have demonstrated up to a **300% gain** in correctness over baseline encoder-decoder models when handling knowledge-intensive tasks.

## 2.4 Retrieval-Augmented Generation for Recommendation (RAG4Rec)

### 2.4.1 Mechanics of RAG-Enhanced Recommendation

While large language models possess vast reasoning capabilities, they are inherently limited by their training data cutoff. In the dynamic environment of e-commerce, news, and streaming services, information changes by the minute. Retrieval-Augmented Generation (RAG) has emerged as the standard production architecture to bridge this gap.

A RAG-based recommendation system (RAG4Rec) operates in a tightly coupled retrieval-generation pipeline. When a user interacts with the system, it first queries a live database — which may include vector stores, enterprise knowledge bases, or real-time inventory systems — to find relevant, up-to-date context. This retrieved context is then appended to the user's prompt, providing the LLM with a factual basis for its generation. RAG is significantly more cost-efficient and scalable than frequent model fine-tuning, as it allows for instant knowledge updates without expensive retraining cycles.

### 2.4.2 Set-Wise Selection vs. Independent Ranking

A critical innovation in 2025 RAG architectures is the shift from "ranking" to "set-wise selection" (SetR). Traditional retrieval systems rank individual results by their independent relevance to a query. However, complex recommendation tasks often require a curated set of items that are collectively diverse, complete, and comprehensive.

The SetR approach uses Chain-of-Thought (CoT) reasoning to explicitly identify the multi-faceted information requirements of a query. It then selects an optimal subset of passages that collectively satisfy all requirements, ensuring the final generation is not just relevant but also comprehensive.

**RAG Innovations for Recommendation Systems**

| RAG Innovation | Description | Primary Benefit |
|---|---|---|
| Set-Wise Selection (SetR) | Optimizes the retrieved set as a whole via CoT reasoning | Higher answer correctness; reduced redundancy |
| Multi-Hop RAG | Iteratively decomposes complex questions into sub-queries | Synthesis of disparate facts for reasoning |
| Structured RAG | Constrains retrieval to verified, domain-specific corpora | 30%–40% reduction in hallucination rates |
| IM-RAG | Employs iterative retrieval refinement during generation | Improved faithfulness in evolving domains |

## 2.5 Explainable AI: From Black Box to Transparent Recommendations

### 2.5.1 Natural Language Explanations and User Trust

One of the most profound changes in the user experience of recommendation systems is the move toward explainability. Older systems, such as those based on Matrix Factorization or deep ranking networks, offered no transparency. LLMs change this by providing natural language explanations that bridge the gap between user intent and system behavior.

Modern recommenders can now provide sophisticated justifications: *"I recommended this hiking boot because you specifically mentioned ankle support in your search query last week, and this brand recently updated their design to address that feedback in user reviews."*

Companies are increasingly using LLMs to assess the quality of recommendations against high-level human principles. Pinterest, for example, tracks "whole-page relevance" using fine-tuned open-source LLMs that evaluate results based on a 5-level semantic guideline.

**Pinterest's Whole-Page Relevance Evaluation Framework**

| Guidance Level | Label | Evaluation Criteria |
|---|---|---|
| L5 | Highly Relevant | Direct semantic match with clear user intent fulfillment |
| L4 | Relevant | Strong thematic alignment with the query and intent |
| L3 | Marginally Relevant | Partial match; potentially useful but not a primary choice |
| L2 | Irrelevant | Weak or tangential connection to the user's inquiry |
| L1 | Highly Irrelevant | No connection; likely a system error or mismatch |

### 2.5.2 Reasoning Graphs and Knowledge Graph Integration

To enable deeper reasoning, modern architectures are integrating Knowledge Graphs (KGs) with LLMs. KGs provide structured, domain-specific knowledge about item attributes and relationships. By projecting graph-structured entity embeddings into the LLM's semantic space via adapter modules, systems can perform "cross-modal reasoning."

Newer models build "reasoning graphs" of user interests, allowing the AI to traverse paths such as: *"User likes Sci-Fi" → "User likes Dystopian Futures" → "User might like this specific Cyberpunk game."* These reasoning-aware systems can uncover and interpret implicit user intent far more effectively than traditional user profiling.

## 2.6 Security and Reliability

### 2.6.1 Hallucination Mitigation Strategies

As recommendation systems become more autonomous and conversational, hallucinations — the generation of factually incorrect or unsupported content — pose a significant challenge. Research in 2025 has reframed hallucinations as a systemic incentive problem, noting that standard training objectives often reward "confident guessing" over "calibrated uncertainty."

Key mitigation strategies include:

- **Reward Models for Calibrated Uncertainty**: New reinforcement learning schemes (e.g., Rewarding Doubt) penalize both over- and underconfidence, encouraging the model to abstain or express uncertainty when evidence is thin.
- **Span-Level Verification**: Generated claims are matched against retrieved evidence at the sentence or phrase level; unsupported claims are flagged or rewritten.
- **Self-Aware Monitoring**: Techniques such as Cross-Layer Attention Probing (CLAP) train classifiers on a model's own internal activations to flag likely hallucinations in real-time.
- **Adversarial Probing (APASI)**: The Adaptive Probing via Self-Instruction method reduces object-level hallucinations by over 10 points on benchmarks.

### 2.6.2 Adversarial Security and Data Integrity

The move toward RAG-based architectures has introduced new vulnerabilities, such as "Data Poisoning" and "Backdoor Attacks." Research into BadRAG has demonstrated that poisoning as little as **0.04%** of a retrieval corpus can lead to a **98.2%** attack success rate. Another threat, TrojanRAG, embeds backdoors directly into retrieval embeddings, bypassing traditional text-based sanitization.

Defensive measures in 2026 include cryptographic document signing and adversarial filtering at inference time, along with proactive "Sentinel" architectures — neuro-symbolic governance layers that monitor agent actions and tool calls for violations of safety or business logic.

## 2.7 Infrastructure Paradigms: From Microservices to Modular Monoliths

### 2.7.1 Actor Frameworks and Shared Memory Blocks

The high-throughput and low-latency requirements of modern agentic recommenders have necessitated a radical shift in infrastructure design. The "Network Tax" associated with traditional microservices — the latency spikes caused by serializing 1MB+ context windows into JSON across multiple HTTP hops — has become a major bottleneck.

Industry leaders are moving toward "Modular Monoliths" using Actor Frameworks such as Ray, Akka, or Erlang/Elixir. Different agents are co-located within the same computational environment and communicate through Shared Memory Blocks (e.g., Plasma Store). This "Zero-Copy Context" approach allows Agent B to read Agent A's thought process instantly by passing a pointer to a memory block rather than a serialized data packet.

### 2.7.2 Inventory Segmentation and Hardware-Aware Serving

Pinterest's serving stack provides a concrete example of re-architecting for next-generation models:

- **Segment 1 (High-Value Inventory)**: ~1 million documents contributing to a majority of revenue. Features are bundled directly inside the PyTorch model file as registered buffers, living on the GPU's high-bandwidth memory (HBM), completely eliminating network overhead.
- **Segment 2 (Long-Tail Inventory)**: For the remaining ~1 billion documents, a high-performance Key-Value store with in-host caching is used to fetch features efficiently.

## 2.8 The Convergence of Search and Recommendation

### 2.8.1 Unified User Representations

The year 2025 has seen a significant move toward unified user representations transferable across different tasks (search, ranking, generative discovery). Spotify has built a large-scale framework for generalized user representations where each listener is mapped to a high-dimensional embedding within a stable vector space. These embeddings serve as the foundation for task-specific models while sharing a common semantic understanding of the user.

### 2.8.2 Multi-Task Bi-Encoders and Task-Agnostic IDs

A central challenge in unification is that Semantic IDs optimized for search (query-item matching) often fail to generalize to recommendation tasks (item-item co-occurrence). The solution has been the development of "Multi-task Bi-encoders" trained via contrastive learning on both query-item pairs and item-item pairs. By clustering these unified embeddings and discretizing them via RQ-KMeans, Semantic IDs are created that lie on the Pareto frontier, providing a balanced solution that enhances both search and recommendation performance.

**Evolution: Pre-2023 vs. 2026 Unified Architectures**

| Feature | Pre-2023 Architectures | 2026 Unified Architectures |
|---|---|---|
| User Representation | Task-specific embeddings | Unified, stable vector space (Autoencoders) |
| Identifier Type | Atomic Item IDs (Random) | Multi-task Semantic IDs (RQ-KMeans) |
| Data Modality | Primarily tabular/text | Multimodal (Audio, Image, Video, Text) |
| Optimization Goal | Task-specific loss (e.g., Search MSE) | Joint contrastive multi-task learning |
| Infrastructure | Separate funnels for Search/Rec | Unified Generative S&R Model |

## 2.9 Depth Scaling vs. Context Scaling: The Expressivity Debate

As models grow to handle longer histories and more complex agentic reasoning, a debate has emerged regarding the most effective axis for model scaling. While systems like Gemini 1.5 have demonstrated the power of long-context windows (1MB+) for in-context learning, newer research from ByteDance (the Keel architecture, January 2026) suggests that depth remains the superior axis for "fundamental expressivity."

The Keel architecture addresses the instability of deep Transformers by reviving the Post-LayerNorm formulation and utilizing "Highway-style" connections, allowing stable training at depths exceeding 1,000 layers. The argument is that while long context allows models to process more data, it does not inherently unlock the capacity for the complex, hierarchical reasoning required for mathematical, coding, and high-level goal planning in agentic recommenders.

## 2.10 Conclusion: Strategic Outlook for 2026 and Beyond

The shift from "Retrieval → Ranking" to "Generative and Agentic" architectures is not merely a change in the model engine; it is a fundamental redefinition of how organizations operationalize knowledge and interact with users.

Key strategic imperatives:

- Develop high-fidelity, real-time **RAG pipelines** for temporal grounding.
- Investigate automation of agentic designs through **supernet frameworks** (MaAS).
- Transition toward **explainable, survey-grounded content quality** to align AI with human values.
- Adopt **modular monolith infrastructure** with hardware-aware serving for production scale.
- Embrace the shift from model-centric to **data-centric AI**, where the quality of retrieval, the integrity of the semantic index, and the robustness of the agentic workflow determine competitive advantage.

---

# References

## Traditional Recommender Systems References

[1] http://ijcai13.org/files/tutorial_slides/td3.pdf

[2] [Hulu](https://web.archive.org/web/20170406065247/http://tech.hulu.com/blog/2011/09/19/recommendation-system.html)

[3] Xavier Amatriain and Justin Basilico. [System Architectures for Personalization and Recommendation](https://link.medium.com/PaefDwO9bab) (by Netflix Technology Blog)

[4] https://github.com/mandeep147/Amazon-Product-Recommender-System

[5] https://github.com/smwitkowski/Amazon-Recommender-System

[6] Recommendation Systems: A Review https://towardsdatascience.com/recommendation-systems-a-review-d4592b6caf4b

[7] [Recommender system using Bayesian personalized ranking](https://towardsdatascience.com/recommender-system-using-bayesian-personalized-ranking-d30e98bba0b9)

[8] [Introduction to Recommender Systems in 2019](https://tryolabs.com/blog/introduction-to-recommender-systems/)

[9] [Introduction to recommender systems](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)

[10] https://en.wikipedia.org/wiki/Recommender_system#Mobile_recommender_systems

[11] Book: 推荐系统实践 项亮

[12] Book: Recommender systems: An introduction, Dietmar Jannach / Markus Zanker / Alexander Felfernig / Gerhard Friedrich, 2013

[14] [Robin Burke Recommender Systems: An Overview](https://www.researchgate.net/publication/220604600_Recommender_Systems_An_Overview)

[15] https://github.com/microsoft/recommenders

[16] https://github.com/kevinmcmahon/tagger

[17] https://github.com/timlyo/pyTag

[18] [Hybrid Recommendation Approaches](https://www.math.uci.edu/icamp/courses/math77b/lecture_12w/pdfs/Chapter%2005%20-%20Hybrid%20recommendation%20approaches.pdf)

[19] https://towardsdatascience.com/mixed-recommender-system-mf-matrix-factorization-with-item-similarity-based-cf-collaborative-544ddcedb330

[20] https://towardsdatascience.com/the-best-forecast-techniques-or-how-to-predict-from-time-series-data-967221811981

[21] [Trend or No Trend: A Novel Nonparametric Method for Classifying Time Series](https://dspace.mit.edu/handle/1721.1/85399)

[22] https://github.com/microsoft/recommenders/blob/master/examples/00_quick_start/als_movielens.ipynb

[23] https://github.com/microsoft/recommenders/tree/master/examples/03_evaluate

[24] Asela Gunawardana and Guy Shani: A Survey of Accuracy Evaluation Metrics of Recommendation Tasks

[25] Dimitris Paraschakis et al, "Comparative Evaluation of Top-N Recommenders in e-Commerce: An Industrial Perspective", IEEE ICMLA, 2015, Miami, FL, USA.

[26] Yehuda Koren and Robert Bell, "Advances in Collaborative Filtering", Recommender Systems Handbook, Springer, 2015. Chris Bishop, "Pattern Recognition and Machine Learning", Springer, 2006.

[27] Errico, James H., et al. "Collaborative recommendation system." U.S. Patent No. 8,949,899. 3 Feb. 2015.

[28] Davidson, James, et al. "The YouTube video recommendation system." Proceedings of the fourth ACM conference on Recommender systems. 2010.

[29] Adomavicius, Gediminas, and Alexander Tuzhilin. "Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions." IEEE transactions on knowledge and data engineering 17.6 (2005): 734–749.

[30] James Loy, [Deep Learning based Recommender Systems, A gentle introduction to modern movie recommenders](https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e)

[31] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua. [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). arXiv:1708.05031. 2017

[32] [Introduction to recommender systems](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada), Overview of some major recommendation algorithms.

[33] [Recommender Systems: From Filter Bubble to Serendipity](https://dorukkilitcioglu.com/2018/10/09/recommender-filter-serendipity.html)

[34] [REFORE: A recommender system for researchers based on bibliometrics](https://www.sciencedirect.com/science/article/abs/pii/S1568494615001258)

[35] [News Recommendation Competition](https://msnews.github.io/competition.html)

## LLM/AI-Era Recommender Systems References

[36] Rajput, S., et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023 (TIGER framework).

[37] Hua, W., et al. "How to Index Item IDs for Recommendation Foundation Models." 2023 (Semantic ID research).

[38] Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020 (foundational RAG).

[39] Gao, Y., et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997. 2024.

[40] Li, L., et al. "A Survey on Large Language Models for Recommendation." arXiv:2305.19860. 2024.

[41] Wang, L., et al. "A Survey on Large Language Model based Autonomous Agents." arXiv:2308.11432. 2024.

[42] Xi, Z., et al. "The Rise and Potential of Large Language Model Based Agents: A Survey." arXiv:2309.07864. 2024.

[43] Zhang, Y., et al. "Multi-agent Architecture Search." arXiv. 2025 (MaAS framework).

[44] Xiao, S., et al. "BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models." arXiv:2406.00083. 2024.

[45] Pinterest Engineering. "Whole-Page Relevance Evaluation with LLMs." Pinterest Engineering Blog, 2025.

[46] Spotify Research. "Generalized User Representations for Multi-task Learning." Spotify Research, 2025.

[47] ByteDance AI Lab. "Keel: Scaling Transformers to 1000+ Layers with Highway-Style Connections." arXiv, January 2026.

[48] Chen, J., et al. "HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models." arXiv:2409.12740 (2024) & "HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation." arXiv:2508.18118 (2025). [GitHub Repo](https://github.com/bytedance/HLLM)
