# Deep Semantic Talent Retrieval Engine

A hybrid candidate retrieval and ranking system designed to match job descriptions with relevant candidate profiles using dense semantic retrieval, lexical retrieval, rank fusion, cross-encoder reranking, and business-specific ranking signals.

The system processes a large candidate dataset and produces a ranked list of candidates with supporting information for each recommendation.

## Overview

Traditional candidate search systems often rely heavily on keyword matching. This can fail when a candidate has relevant experience but uses different terminology from the job description.

This project combines multiple retrieval and ranking techniques:

* Dense semantic retrieval using BAAI BGE embeddings
* Lexical retrieval using BM25
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder reranking
* Experience and candidate-availability signals
* Domain-specific positive and negative signals
* Candidate ranking and score normalization
* Structured ranking explanations

The overall pipeline follows a retrieve-then-rerank architecture.

```text
                         Job Description
                                |
                +---------------+---------------+
                |                               |
                v                               v
        Dense Semantic Search              BM25 Search
        BGE Embeddings                    Lexical Retrieval
                |                               |
                +---------------+---------------+
                                |
                                v
                     Reciprocal Rank Fusion
                                |
                                v
                       Candidate Pool
                                |
                                v
                       Cross-Encoder
                         Reranking
                                |
                                v
                    Business / Domain Signals
                                |
                                v
                       Final Ranking
                                |
                                v
                     Explainability Layer
                                |
                                v
                       Ranked Candidates
```

## Problem Statement

Given a job description and a large collection of candidate profiles, the system needs to identify candidates who are relevant to the role based on:

1. Semantic similarity to the job description
2. Exact skill and terminology matches
3. Contextual relevance of the candidate's background
4. Relevant professional experience
5. Candidate availability signals
6. Domain-specific relevance

A single retrieval technique is insufficient for this problem.

Keyword-based retrieval can miss semantically related candidates, while embedding-based retrieval can sometimes overlook exact technical terms or explicit domain requirements.

The system therefore combines lexical and semantic retrieval before applying a more expensive reranking model.

## Dataset

The system was developed against a dataset containing approximately 100,000 candidate profiles.

Candidate records contain structured information including:

```text
candidate_id
profile
career_history
education
skills
certifications
languages
redrob_signals
```

The `profile` information includes fields such as:

```text
headline
summary
location
years_of_experience
current_title
current_company
```

Candidate signals include information such as:

```text
open_to_work_flag
recruiter_response_rate
github_activity_score
notice_period_days
preferred_work_mode
expected_salary_range_inr_lpa
```

## Candidate Representation

Structured candidate information is converted into a textual representation for retrieval.

The initial retrieval document contains:

```text
Candidate Headline

Candidate Summary

Skills:
skill_1, skill_2, skill_3, ...
```

This representation is used for both dense semantic retrieval and BM25 lexical retrieval.

For cross-encoder reranking, a richer representation is constructed containing:

```text
Title
Summary
Years of Experience
Skills
Career History
```

## System Architecture

### 1. Candidate Data Loading

Candidate profiles are loaded from JSONL records.

Each line represents one candidate and is parsed into a Python dictionary.

```python
for line in file:
    data.append(json.loads(line))
```

The implementation processes 100,000 candidate records.

### 2. Dense Semantic Retrieval

The project uses:

```text
BAAI/bge-small-en-v1.5
```

through the Sentence Transformers library.

Candidate embeddings are generated offline and stored for reuse.

At query time, only the job description needs to be encoded.

```text
Candidate Profiles
        |
        v
BGE Embedding Model
        |
        v
Candidate Embeddings
        |
        v
Stored Embedding Matrix
```

The job description is then encoded into an embedding:

```python
jd_embedding = model.encode([text])
```

Cosine similarity is used to compare the job-description embedding with candidate embeddings.

```python
similarities = cosine_similarity(
    jd_embedding,
    docs_embeddings
)
```

This produces a semantic similarity score for each candidate.

### 3. Lexical Retrieval with BM25

The system also uses BM25 through BM25S.

```python
corpus = bm25s.tokenize(documents)

bm25 = bm25s.BM25()
bm25.index(corpus)
```

The job description is tokenized and used as the query.

```python
query = bm25s.tokenize(text)

results500, scores500 = bm25.retrieve(
    query,
    corpus=documents,
    k=500
)
```

BM25 provides lexical matching based on terms appearing in the query and candidate documents.

This is useful for explicit technologies, tools, frameworks, and terminology.

### 4. Hybrid Retrieval

Dense retrieval and BM25 capture different types of relevance.

| Retrieval Method | Primary Strength       |
| ---------------- | ---------------------- |
| Dense Retrieval  | Semantic similarity    |
| BM25             | Exact lexical matching |

For example, dense retrieval may recognize that:

```text
"neural information retrieval"
```

is related to:

```text
"semantic search"
```

while BM25 can strongly reward an exact occurrence of a required technology such as:

```text
"PyTorch"
```

Combining both retrieval methods improves the diversity of candidates entering the reranking stage.

### 5. Reciprocal Rank Fusion

The rankings produced by dense retrieval and BM25 are combined using Reciprocal Rank Fusion.

The implementation uses:

```python
def run_rrf(list1, list2, k=60):
```

The RRF contribution for a document is:

$$
RRF(d) =
\sum_r \frac{1}{k + rank_r(d)}
$$

where:

* `d` is a candidate
* `r` represents a retrieval system
* `rank_r(d)` is the candidate's rank in that system
* `k` is the ranking constant

The project uses:

```text
k = 60
```

RRF is useful because BM25 and dense retrieval produce scores on different scales. Instead of directly adding incompatible raw scores, RRF combines the rankings themselves.

### 6. Cross-Encoder Reranking

After the initial retrieval stage, candidate-job-description pairs are passed to a Cross-Encoder.

The project uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

A bi-encoder independently represents the query and candidate:

```text
Job Description -> Vector
Candidate        -> Vector

Vector Similarity
```

A Cross-Encoder instead evaluates the query and candidate together:

```text
Job Description
       +
Candidate Profile
       |
       v
Cross-Encoder
       |
       v
Relevance Score
```

This allows the model to capture more detailed interactions between the job description and candidate profile.

The tradeoff is computational cost.

Therefore, the Cross-Encoder is used after the initial retrieval stage rather than against the complete candidate dataset.

This follows the standard:

```text
Retrieve -> Rerank
```

architecture.

### 7. Candidate Heuristic Scoring

The project combines semantic similarity with additional candidate signals.

The initial heuristic score is calculated using:

```python
score = similarity * 100
```

Additional signals include:

* Experience within a target range
* Open-to-work status
* Recruiter response rate

The implementation awards additional score for candidates with:

```text
5 to 9 years of experience
```

and for candidates with an active open-to-work signal.

Recruiter response rate is also incorporated into the score.

### 8. Domain Validation

A rule-based domain validation layer provides explicit positive and negative signals.

Candidate title, career history, and skills are combined into a searchable text representation.

Positive signals include terms such as:

```text
machine learning
machine learning
AI engineer
artificial intelligence
NLP
search
retrieval
ranking
recommendation
vector
embedding
LLM
RAG
transformer
sentence transformer
FAISS
Pinecone
Qdrant
Weaviate
Milvus
BM25
cross encoder
information retrieval
```

Negative signals include roles such as:

```text
mechanical engineer
civil engineer
customer support
sales executive
marketing
content writer
accountant
HR manager
operations manager
business analyst
project manager
graphic designer
UI designer
UX designer
```

These signals are used as additional ranking features rather than replacing semantic retrieval.

### 9. Final Ranking

The final ranking combines three major components:

```python
final_score = (
    normalized_cross * 0.6 +
    heuristic_score * 0.4 +
    domain_score
)
```

Conceptually:

$$
FinalScore =
0.6 \times CrossEncoderScore
+
0.4 \times HeuristicScore
+
DomainScore
$$

The Cross-Encoder receives the highest weighting because it provides contextual query-candidate relevance.

The heuristic and domain components provide additional business and domain-specific information.

### 10. Score Normalization

Cross-Encoder scores are passed through a sigmoid function:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The result is converted to a 0-100 style scale.

Final candidate scores are subsequently normalized using min-max normalization:

$$
x' =
\frac{x-x_{min}}
{x_{max}-x_{min}}
$$

This produces normalized scores between 0 and 1 for the final submission.

## Explainability

The project contains a `CandidateExplainer` component that generates structured explanations for ranked candidates.

The explanation contains:

```text
Candidate ID
Candidate headline
Final composite score
Cross-Encoder score
Vector similarity
RRF score
Key skills
Experience
Open-to-work status
Recruiter response rate
GitHub activity
Notice period
Preferred work mode
Expected salary range
```

Example output structure:

```json
{
  "candidate_id": "CAND_0000000",
  "headline": "Senior Machine Learning Engineer",
  "final_composite_score": 38.84,
  "score_breakdown": {
    "Semantic Alignment (Cross-Encoder)": "...",
    "Vector Semantic Match (Bi-Encoder)": "...",
    "RRF Fusion Score": "...",
    "Experience & Signal Tier": "High Fit"
  },
  "key_skills": [
    "Information Retrieval",
    "Machine Learning",
    "RAG"
  ],
  "justification_bullets": [
    "Strong contextual alignment",
    "Relevant experience",
    "Actively looking"
  ]
}
```

The explainability layer is rule-based and exposes the signals that contributed to the ranking rather than attempting to explain the internal reasoning of the neural models.

## Performance Considerations

The system separates expensive processing from query-time processing wherever possible.

### Offline processing

Candidate embeddings can be generated and stored ahead of time:

```text
Candidate Profiles
       |
       v
Embedding Model
       |
       v
candidate_embeddings.npy
```

### Query-time processing

Only the job description needs to be embedded:

```text
Job Description
       |
       v
Embedding Model
       |
       v
Similarity Search
```

This avoids repeatedly encoding the complete candidate dataset for every query.

The Cross-Encoder is also applied after retrieval rather than across the entire dataset, reducing the number of expensive transformer inference operations.

## Technologies Used

### Machine Learning and NLP

* Python
* PyTorch
* Sentence Transformers
* Hugging Face Transformers
* BAAI BGE embeddings
* Cross-Encoder
* SciPy
* scikit-learn

### Information Retrieval

* BM25S
* Dense Vector Retrieval
* Cosine Similarity
* Reciprocal Rank Fusion
* Cross-Encoder Reranking

### Data Processing

* JSONL
* NumPy
* Pandas
* Python-docx

### Development Environment

* Jupyter Notebook
* Python 3.11

## Project Structure

```text
Deep-Semantic-Talent-Retrieval-Engine/
|
├── main.ipynb
├── main.py
├── submission.csv
├── candidate_embeddings.npy
├── cross_encoder_scores.npy
├── Idea Submission Template _ Redrob.pptx
└── [PUB] India_runs_data_and_ai_challenge/
```

`main.ipynb` contains the primary implementation and experimentation pipeline.

`submission.csv` contains the generated ranked candidate output.

The embedding and Cross-Encoder score files are used to avoid repeatedly performing expensive model inference during experimentation.

## End-to-End Pipeline

The complete pipeline can be summarized as:

```text
1. Load candidate dataset
          |
2. Convert candidate profiles to text
          |
3. Generate or load candidate embeddings
          |
3. Encode job description
          |
4. Dense cosine similarity
          |
5. BM25 lexical retrieval
          |
6. Convert retrieval results to candidate IDs
          |
7. Reciprocal Rank Fusion
          |
8. Build richer candidate representations
          |
9. Cross-Encoder reranking
          |
10. Apply experience and behavioral signals
          |
11. Apply domain validation rules
          |
12. Calculate final score
          |
13. Normalize scores
          |
14. Generate explanations
          |
15. Export ranked candidates to CSV
```

## Why This Architecture?

A single retrieval technique has limitations.

### Dense Retrieval

Provides semantic matching but may not always prioritize exact technical terminology.

### BM25

Provides strong lexical matching but may miss semantically related terminology.

### RRF

Combines independent retrieval rankings without requiring their raw scores to be directly comparable.

### Cross-Encoder

Provides more precise query-candidate relevance modeling but is computationally more expensive.

### Business and Domain Signals

Allow the ranking system to incorporate information that pure semantic relevance does not capture.

Together, these components form a multi-stage retrieval and ranking pipeline.

## Limitations

The current implementation is primarily an experimental and competition-oriented retrieval pipeline.

Several areas could be improved for a production deployment:

### Approximate Nearest Neighbor Search

The current dense retrieval stage compares the query embedding against the candidate embedding matrix.

For significantly larger datasets, an ANN index such as FAISS or HNSW could reduce retrieval latency.

### Search Infrastructure

BM25 retrieval could be moved into a dedicated search engine such as Elasticsearch or OpenSearch for distributed indexing and production-scale querying.

### Learned Ranking

The final ranking weights are currently manually defined.

A learning-to-rank model could learn the optimal combination of:

```text
Dense similarity
BM25 relevance
Cross-Encoder score
Experience
Skill overlap
Availability
Recruiter response rate
Other candidate signals
```

Possible approaches include:

* LambdaMART
* XGBoost ranking
* LightGBM ranking
* Neural learning-to-rank models

### Evaluation

A production system should be evaluated using ranking metrics such as:

* Precision@K
* Recall@K
* MRR
* NDCG

and ideally against human-labeled candidate relevance data.

### Score Calibration

The current ranking combines several manually weighted signals with different numerical scales. A production implementation could introduce systematic feature normalization and learned weighting.

## Future Improvements

Potential improvements include:

1. Replace brute-force vector similarity with an ANN index.
2. Use a dedicated lexical search engine for BM25 retrieval.
3. Introduce a learned ranking model.
4. Add systematic relevance evaluation using Precision@K, Recall@K, MRR and NDCG.
5. Add candidate skill-overlap and job-requirement features.
6. Improve score calibration across retrieval and ranking stages.
7. Add a production API for recruiter queries.
8. Add caching for repeated job-description searches.
9. Introduce batch and distributed Cross-Encoder inference.
10. Add monitoring for retrieval quality and ranking drift.

## Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Information Retrieval
* Semantic Search
* Dense Embeddings
* Bi-Encoders
* Cross-Encoders
* BM25
* Vector Similarity
* Reciprocal Rank Fusion
* Retrieve-Then-Rerank Architectures
* Feature Engineering
* Heuristic Ranking
* Score Normalization
* Explainable Ranking
* Offline Embedding Generation
* Candidate Retrieval at Scale

## Author

**Hari Nivedhan P**

B.Tech Information Technology
Saveetha Engineering College

GitHub: [hari7niv](https://github.com/hari7niv)

LinkedIn: [Hari Nivedhan](https://linkedin.com/in/hari-nivedhan)
