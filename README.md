# Lab 3: Contextual Bandit-Based News Article Recommendation


**Student:** Kunal Ranjan  
**Roll Number:** U20230022  
**Branch:** `Kunal_U20230022`

---

## Project Report

This project implements a **Contextual Multi-Armed Bandit (CMAB)** system for personalized news article recommendation. Users are classified into one of three contexts (User1, User2, User3), and the bandit learns which news category (Entertainment, Education, Tech, Crime) maximizes reward for each context.

---

## Approach & Design Decisions

### 1. User Classification (Context Prediction)

Three classifiers were compared on a 80/20 stratified validation split:

| Model               | Accuracy |
|---------------------|----------|
| Decision Tree       | 83.50%   |
| **Logistic Regression** | **92.00%** |
| Random Forest       | 89.75%   |

**Logistic Regression** was selected as the context classifier due to the highest validation accuracy. Features include age, income, clicks, purchase_amount, subscriber status, and one-hot encoded browser_version and region_code (134 features total). StandardScaler is applied before classification.

### 2. Contextual Bandit – Arm Mapping

| Arm Index (j) | News Category                          | User Context |
|---------------|----------------------------------------|--------------|
| 0–3           | Entertainment, Education, Tech, Crime  | User1        |
| 4–7           | Entertainment, Education, Tech, Crime  | User2        |
| 8–11          | Entertainment, Education, Tech, Crime  | User3        |

Rewards are sampled via the provided `rlcmab_sampler` package initialized with roll number `i=22`.

### 3. Bandit Algorithms & Hyperparameter Sweeps

Each algorithm was swept over multiple hyperparameter values (T=10,000 steps) to compare expected payoffs:

- **Epsilon-Greedy** — ε ∈ {0.01, 0.1, 0.3, 0.5}
- **UCB** — C ∈ {0.5, 1.0, 2.0, 5.0}
- **SoftMax** — τ ∈ {0.1, 0.5, 1.0, 5.0}

### 4. Recommendation Engine

The best bandit policy (UCB, C=2.0) and the Logistic Regression classifier are combined into an end-to-end pipeline: classify user → select optimal category via learned Q-values → sample a random article from that category.

---

## Key Results

### Best Hyperparameters

| Algorithm    | Best Hyperparameter | Avg Reward |
|--------------|---------------------|------------|
| ε-Greedy     | ε = 0.01            | 5.6316     |
| **UCB**      | **C = 2.0**         | **5.7134** |
| SoftMax      | τ = 0.1             | 5.7079     |

### Optimal Arms Per Context

| Context | Best Category | Q-value |
|---------|---------------|---------|
| User1   | Crime         | ~5.47   |
| User2   | Education     | ~3.77   |
| User3   | Crime         | ~7.95   |

### Key Observations

1. **UCB (C=2.0)** achieves the highest overall average reward. It explores principled via confidence bounds and converges reliably.
2. **SoftMax (τ=0.1)** is nearly as good but is essentially greedy — it converges fast but explores very little.
3. **ε-Greedy** is most sensitive to hyperparameter choice; ε=0.01 performs well but ε=0.5 drops significantly.
4. Each user context has a **distinct optimal category**, validating the contextual approach over a context-free bandit.

---

## Reproducing the Experiments

### Prerequisites

- Python 3.11+
- Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- The provided `rlcmab_sampler` package (included in the repository)


### Data Files

- `data/train_users.csv` — labeled user data for classifier training
- `data/test_users.csv` — unlabeled user data for recommendations
- `data/news_articles.csv` — news articles with category labels

---

