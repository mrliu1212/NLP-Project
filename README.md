# Media Bias Detection through Topic Clustering and Outlet-Specific Language Models

## Overview

This project explores how different media outlets semantically and emotionally frame global issues. It aims to identify variations in coverage, tone, and topic emphasis, and investigates whether language models can detect and characterize such media bias.

Our approach is twofold:
1. **Topic-Level Framing and Tone Analysis**
2. **Outlet-Specific Language Modeling**

We analyze news articles from CNN, BBC, Fox News, and Global Times, grouped by topic and collected via NewsAPI.org. Each sentence is treated as a standalone document for fine-grained analysis.

## Part 1: Clustering and Framing Analysis

### Clustering (TF-IDF + K-Means / LDA)

**Goal**: Identify framing patterns by clustering articles within the same topic.  
**Method**:
- Convert articles to TF-IDF vectors.
- Apply K-Means or LDA to group them into subtopics or narrative frames.
- Label clusters using top keywords.
- Visualize with dimensionality reduction techniques (e.g., t-SNE).

**Outcome**: Reveals framing diversity across outlets for the same topic.

### Omission Analysis

**Goal**: Detect which outlets omit coverage of certain subtopics.  
**Method**:
- Track outlet distribution across discovered clusters.
- Normalize by total articles per outlet.
  
**Outcome**: Quantifies selective coverage, forming the basis for omission-based bias assessment.

### Tone Analysis

**Goal**: Assess emotional and ideological tone per outlet-topic pair.  
**Method**:
- Use sentiment analysis tools (VADER, TextBlob) to compute polarity and subjectivity scores.
- Analyze lexical features like modal verbs, emotional adjectives, and hedging phrases.
  
**Outcome**: Detects tonal bias (e.g., optimistic vs. alarmist framing of AI).

## Part 2: Outlet-Specific Language Modeling

### Generative Modeling (Outlet-Specific LMs)

**Goal**: Compare stylistic framing by generating answers to the same prompts.  
**Method**:
- Train/fine-tune a generative LM (e.g., GPT-2) for each outlet.
- Evaluate outputs for prompts like:
  - “What is climate change?”
  - “How should governments respond to protests?”

**Outcome**: Illustrates ideological framing and language style differences.

### Predictive Modeling (Source Classifier)

**Goal**: Predict the outlet of an article based on text.  
**Method**:
- Train a classifier (e.g., BERT, logistic regression) on labeled sentences or articles.
- Evaluate using accuracy and confusion matrix.

**Outcome**: Measures how linguistically distinguishable outlets are. Misclassifications can indicate neutrality or similarity in tone.

## Dataset

- **Sources**: CNN, BBC, Fox News, Global Times
- **Collection**: NewsAPI.org
- **Preprocessing**: Sentence splitting, topic grouping, stopword removal, TF-IDF vectorization

## Evaluation

- **Clustering**: Topic coherence, outlet distribution balance
- **Tone**: Average sentiment per outlet-topic
- **Classification**: Accuracy, confusion matrix, stylistic separability
- **Generative Models**: Qualitative and lexical comparison of generated outputs

## Folder Structure

```
media-bias/
├── data/                  # News articles and metadata grouped by outlet and topic
├── clustering/            # TF-IDF, K-Means, LDA implementations
├── sentiment/             # Sentiment and tone analysis tools
├── generative_models/     # Fine-tuned LMs and prompt evaluation
├── classifier/            # Source prediction model
├── notebooks/             # Interactive notebooks for analysis and visualization
└── README.md              # Project documentation
```

## Dependencies

- Python 3.8+
- scikit-learn
- pandas, numpy
- gensim
- nltk
- transformers (HuggingFace)
- matplotlib, seaborn
- TextBlob / VADER

Install using:
```bash
pip install -r requirements.txt
```

## Authors
Developed by Group 7 for the Machine Learning course at Bocconi University.
