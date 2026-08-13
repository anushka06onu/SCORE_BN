# SCORE-BN: Severity-Consistent Ordinal Robustness for Bangla Healthcare Query Classification Across Native, Romanized, and Code-Mixed Text

A shorter course-project title: **Robust Severity Classification of Real-World Bangla Healthcare Queries Using NLP**

## Table of Contents
- [Project Title & Description](#project-title--description)
- [The Research Problem](#the-research-problem)
- [Real-world Textual Dataset](#real-world-textual-dataset)
- [Methodological Novelty (SCORE-BN)](#methodological-novelty-score-bn)
- [Research Objectives & Questions](#research-objectives--questions)
- [Results Audit](#results-audit)
- [Literature Review & Gap Analysis](#literature-review--gap-analysis)

## Project Title & Description
The project is a multiclass ordinal text-classification problem. It must be presented as an experimental healthcare-query prioritisation system—not a medical diagnostic system.

## The Research Problem
The system receives a real-world healthcare query written in Bangla and predicts one of four ordered categories: 
`General < Routine < Urgent < Emergency`

The central research question is:
> Can a healthcare severity classifier maintain the same correct severity prediction when a Bangla query is written in native Bangla, Romanized Bangla, or Bangla-English code-mixed form?

## Real-world Textual Dataset
The project utilizes the **Bangla Healthcare Severity Dataset**. It satisfies the requirement for real-world public textual data. It is not synthetic.

| Property | Description |
| :--- | :--- |
| **Samples** | 5,263 |
| **Language** | Bangla |
| **Source** | Public Facebook and YouTube healthcare discussions |
| **Data type** | Real-world textual queries |
| **Labels** | Emergency, Urgent, Routine, General Query |
| **Annotation** | Manual |
| **Columns** | Text, Categories, Action Needed |
| **Licence** | CC BY 4.0 |
| **Publication date** | June 26, 2026 |

**Important dataset rules:**
* **Input:** Text
* **Target:** Categories
* *Do not use Action Needed as an input because it may reveal the target.*
* Remove exact and near-duplicate queries before splitting.
* Perform the train/validation/test split before augmentation.
* Confirm that another course group has not selected the same dataset.
* Your core dataset remains completely real-world. Transliteration and code-mixing are controlled training transformations applied after splitting, not replacements for the real data.

## Methodological Novelty (SCORE-BN)
**SCORE-BN** stands for: **Severity-Consistent Ordinal Robustness for Bangla NLP**

The proposed novelty is a severity-preserving cross-script consistency objective integrated with ordinal learning and asymmetric under-prioritisation for real-world Bangla healthcare-query classification.

Its proposed loss function is:
`L_SCORE = L_ordinal + λ1(L_cross-script consistency) + λ2(L_under-prioritisation)`

* **Component 1: Ordinal severity learning:** Instead of treating the four labels as unrelated categories, the model learns three thresholds (Above General, Above Routine, Above Urgent). A CORAL/CORN-style ordinal head is used.
* **Component 2: Cross-script severity consistency:** For a real query `x`, create a meaning-preserving Romanized or code-mixed view `T(x)`. The model should produce similar probability distributions. This penalises the model when the same query receives different severity predictions only because its writing style changed.
* **Component 3: Asymmetric under-prioritisation loss:** Predicting below the correct severity receives an additional penalty. Predicting Emergency as Routine is therefore more costly than confusing Routine with Urgent. The cost weights must be described as experimental values unless they are reviewed by a qualified healthcare professional.

## Research Objectives & Questions

### Research Objectives
1. Benchmark traditional, deep-learning and transformer models on real-world Bangla healthcare severity classification.
2. Measure how native-to-Romanized and code-mixed variation affects model performance.
3. Develop SCORE-BN for severity-consistent cross-script prediction.
4. Reduce under-prioritisation and cross-script prediction changes.
5. Explain which words or phrases influence severity predictions.
6. Develop a research demonstration interface.

### Research Questions (RQs)
* **RQ1:** Which NLP model performs best on real-world Bangla healthcare severity classification?
* **RQ2:** How much does performance decline under Romanization and code-mixing?
* **RQ3:** Does SCORE-BN improve cross-script consistency?
* **RQ4:** Does SCORE-BN reduce under-prioritisation errors?
* **RQ5:** Are its predictions reasonably interpretable?

### Hypotheses
* **H1:** Transformers will outperform traditional models in macro-F1.
* **H2:** Ordinary models will perform worse on Romanized and code-mixed queries.
* **H3:** SCORE-BN will reduce cross-script disagreement.
* **H4:** Asymmetric ordinal training will reduce severe under-prioritisation.

## Results Audit

The 70/15/15 stratified split is correct. Importantly, the Action Needed field was excluded, which prevents obvious target leakage.

### Dataset Verification

| Item | Verified Result |
| :--- | :--- |
| Original rows | 5,263 |
| Missing text | 0 |
| Missing labels | 0 |
| Normalized duplicates detected | 45 |
| **Clean rows** | **5,215** |
| Training | 3,650 |
| Validation | 782 |
| Test | 783 |
| Classes | 4 |
| Vocabulary reported | 4,758 |

### Class Distribution (Reasonably Balanced)

| Class | Samples |
| :--- | :--- |
| General Query | 1,162 |
| Routine | 1,260 |
| Urgent | 1,448 |
| Emergency | 1,345 |

### Verified Test Results

*Note: BanglaBERT is the best model by test Macro-F1. SCORE-BN is the best proposed robustness-oriented model but is not the best overall classifier.*

| Model | Accuracy | Macro-F1 | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **BanglaBERT** | 0.9323 | 0.9347 | 0.9882 |
| **SCORE-BN** | 0.9183 | 0.9218 | 0.9872 |
| Tuned Linear SVM | 0.9170 | 0.9197 | Not saved |
| Linear SVM | 0.9157 | 0.9186 | 0.9841 |
| CNN | 0.9157 | 0.9186 | 0.9870 |
| BiGRU | 0.9029 | 0.9060 | 0.9834 |
| Logistic Regression | 0.9017 | 0.9049 | Missing in final table |
| BiLSTM | 0.9017 | 0.9045 | 0.9779 |
| XGBoost | 0.8991 | 0.9023 | 0.9799 |
| Random Forest | 0.8966 | 0.9003 | 0.9810 |
| Tuned XGBoost | 0.8902 | 0.8935 | Not saved |
| Multinomial NB | 0.8838 | 0.8870 | 0.9808 |

## Literature Review & Gap Analysis
A six-year literature review (2021–2026) revealed that while healthcare text classification, medical triage, Bangla NLP, and transliteration robustness have been studied individually, no indexed paper combines all of the following:

* Real-world Bangla social-media healthcare queries
* Ordered four-level severity classification
* Paired native–Romanized–code-mixed representations
* A training loss that forces severity consistency across those representations
* Asymmetric penalisation of under-prioritisation
* Cross-script robustness evaluation

This is the exact gap the SCORE-BN project addresses.
