# Fake News Detection using Machine Learning

**Author:** Sanjay Kumar Chhetri  
**Date:** December 16, 2025  
**Project:** Misinformation Detection using Natural Language Processing  
**Context:** Springboard Data Science Career Track Capstone Project  
**Context:** Springboard Data Science Career Track Capstone Project

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Literature & Research Context](#literature--research-context)
- [Limitations & Ethics](#limitations--ethics)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## 🎯 Overview

This project implements an end-to-end **Natural Language Processing (NLP)** and **Machine Learning** pipeline to detect misinformation in online news articles. Using the **FakeNewsNet** dataset, the analysis explores linguistic and structural differences between *fake* and *real* news and evaluates multiple supervised learning models for predictive performance.

The project emphasizes:
- Rigorous **exploratory data analysis (EDA)** with statistical validation
- **Data-driven feature selection** with clear justification
- Interpretable machine learning models
- Responsible discussion of **ethical limitations** in automated misinformation detection

**Primary Research Question:**  
*"How do linguistic and structural properties of news articles differ between fake and real news, and how effectively can these differences be leveraged for automated detection?"*

### Project Objectives
1. **Systematic Feature Exploration** - Investigate all potential modeling features with appropriate visualizations
2. **Statistical Validation** - Apply inferential statistics (t-tests, Mann-Whitney U, Chi-square) with effect sizes
3. **Feature Engineering** - Create TF-IDF features informed by EDA and literature
4. **Model Development** - Build and compare multiple classification algorithms
5. **Interpretability** - Analyze feature importance and model decisions

---

## 📊 Dataset

### FakeNewsNet Dataset
- **Source:** [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) (Shu et al., 2018)
- **Size:** 23,196 news articles
  - 16,401 Real news articles
  - 5,323 Fake news articles
- **Domains:** 
  - PolitiFact (political news)
  - GossipCop (entertainment news)
- **Format:** CSV files with article titles and metadata
- **Labels:** Binary classification (0 = Real, 1 = Fake)

**Dataset Characteristics:**
- Multi-domain coverage (politics and entertainment)
- Real-world labels from fact-checking organizations
- Class imbalance (3:1 real to fake ratio)

---

## 🔬 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- **Class Distribution Analysis:** Visualized data imbalance
- **Text Length Analysis:** Word count and character distributions
- **Readability Metrics:** Flesch Reading Ease scores
- **Word Frequency Analysis:** Top n-grams for fake vs real news
- **Statistical Testing:**
  - Independent t-tests for numeric features
  - Mann-Whitney U test (non-parametric validation)
  - Chi-square test for categorical independence
  - Effect size calculations (Cohen's d, Cramér's V)

### 2. **Feature Engineering**
- **TF-IDF Vectorization:**
  - 5,000 features (optimal balance of coverage vs dimensionality)
  - Unigrams + bigrams (captures words and phrases)
  - min_df=5 (removes rare noise)
  - max_df=0.8 (removes overly common terms)
  - English stop words removed

### 3. **Model Training & Evaluation**
Three interpretable machine learning models:
- **Logistic Regression** - Baseline model, interpretable coefficients
- **Random Forest** - Ensemble learning, handles non-linearity
- **Gradient Boosting** - Sequential error correction

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (threshold-independent performance)
- Confusion matrices for error analysis

### 4. **Interpretation & Insights**
- Feature importance analysis
- Top predictive words for fake vs real news
- Error analysis of misclassifications
- Linguistic pattern identification

---

## ✨ Key Features

- **Research-Grounded:** Built on 30+ academic papers in misinformation detection
- **Statistical Rigor:** All findings validated with statistical tests (p < 0.001)
- **Interpretable Models:** Focus on explainable AI over black-box approaches
- **Reproducible:** Complete pipeline from raw data to trained models
- **Comprehensive Documentation:** Detailed explanations and literature context
- **Production-Ready:** Clean code following best practices

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2.git
cd Misinformation-Detection-ML-Model2
```

2. **Install required packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk scipy
```

3. **Download NLTK data (optional, for extended NLP features):**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. **Dataset:**
The FakeNewsNet dataset is already included in the repository under `FakeNewsNet/dataset/`:
- `politifact_fake.csv`
- `politifact_real.csv`
- `gossipcop_fake.csv`
- `gossipcop_real.csv`

---

## 📖 Usage

### Running the Notebook

1. **Open Jupyter Notebook:**
```bash
jupyter notebook misinformation_analysis.ipynb
```

2. **Execute cells sequentially:**
   - The notebook is designed to run top-to-bottom
   - All dependencies are imported in the first code cell
   - Results include visualizations and statistical summaries

### Key Sections:
1. **Introduction** - Background and objectives
2. **Data Acquisition** - Loading CSV files
3. **Data Cleaning** - Handling missing values and duplicates
4. **EDA** - Statistical analysis and visualizations
5. **Feature Engineering** - TF-IDF transformation
6. **Modeling** - Training three ML models
7. **Evaluation** - Performance metrics and comparison
8. **Interpretation** - Feature importance and insights
9. **Limitations & Ethics** - Responsible AI considerations
10. **Conclusion** - Summary and future directions

---

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **83.62%** | 75.12% | 72.45% | 73.76% | **87.83%** |
| Random Forest | 81.20% | 71.89% | 68.32% | 70.06% | 83.57% |
| Gradient Boosting | 80.43% | 70.15% | 66.78% | 68.42% | 78.32% |

**Winner:** Logistic Regression (best ROC-AUC and overall balance)

### 💡 Key Insight

**Logistic Regression performs competitively despite its simplicity.** Linear models are well-suited to sparse TF-IDF representations and benefit from regularization in high-dimensional text spaces. This demonstrates that simpler, well-justified models can achieve robust performance when paired with strong EDA and feature engineering.

### Key Findings

#### Top Predictive Features for Fake News:
- "trump", "report", "breaking", "leaked", "shocking"
- More sensational and emotional language
- Celebrity names and rumors (entertainment domain)
- Political controversy terms (political domain)

#### Top Predictive Features for Real News:
- "season", "awards", "reveals", "birthday", "shares"
- More formal and neutral language
- Event-based terminology
- Temporal markers ("first", "2018", "annual")

### Statistical Significance
- **Text Length:** Significant difference (t-test p < 0.001, Cohen's d = 0.45)
- **Readability:** Fake news slightly more readable (p < 0.001)
- **Source Distribution:** Not independent of label (Chi-square p < 0.001)

---

## 📁 Project Structure

```
Misinformation-Detection-ML-Model2/
│
├── misinformation_analysis.ipynb    # Main analysis notebook
├── README.md                         # This file
│
└── FakeNewsNet/                      # Dataset directory
    ├── dataset/
    │   ├── politifact_fake.csv      # Fake political news
    │   ├── politifact_real.csv      # Real political news
    │   ├── gossipcop_fake.csv       # Fake entertainment news
    │   └── gossipcop_real.csv       # Real entertainment news
    │
    ├── code/                         # Original dataset collection scripts
    └── README.md                     # Dataset documentation
```

---

## 🔍 Key Findings

### Linguistic Patterns Discovered

1. **Sensationalism in Fake News:**
   - Higher use of emotional triggers and loaded language
   - More exclamation points and emphasis
   - Clickbait-style headlines

2. **Formality in Real News:**
   - Professional journalistic terminology
   - Attribution to sources
   - Balanced and neutral reporting style

3. **Domain-Specific Patterns:**
   - Political fake news focuses on controversy and conspiracy
   - Entertainment fake news emphasizes celebrity gossip and rumors
   - Real news in both domains uses more factual, event-based language

4. **Statistical Validation:**
   - All observed differences statistically significant (p < 0.001)
   - Effect sizes (Cohen's d) indicate practical significance
   - Consistent patterns across multiple feature types

---

## 📚 Literature & Research Context

This project is grounded in extensive academic research on misinformation detection:

### Foundational Research
- **Linguistic Features:** Pérez-Rosas et al. (2018), Zhou & Zafarani (2020)
- **Content-Based Approaches:** Shu et al. (2017), Sharma et al. (2019)
- **Machine Learning Models:** Ahmed et al. (2020), Kaliyar et al. (2021)

### Key Methodological Influences
- **TF-IDF Effectiveness:** Granik & Mesyura (2017), Oshikawa et al. (2020)
- **N-gram Selection:** Wang (2017), Castelo et al. (2019)
- **Statistical Validation:** Zhou et al. (2020), Pérez-Rosas & Mihalcea (2015)

### Dataset Benchmark
- **FakeNewsNet:** Shu et al. (2018) - Multi-domain dataset with fact-checked labels

Full references available in the notebook's References section.

---

## ⚠️ Limitations & Ethics

### Technical Limitations
- **Text-Only Analysis:** Ignores images, videos, and multimedia context
- **Domain Specificity:** Trained on political and entertainment news only
- **Temporal Drift:** Language patterns evolve; models require periodic retraining
- **Class Imbalance:** 3:1 real to fake ratio may affect minority class performance

### Ethical Considerations
- **False Positives:** Risk of incorrectly flagging legitimate news
- **False Negatives:** Missing fake news allows misinformation to spread
- **Bias Concerns:** Models may learn source credibility rather than content quality
- **Transparency:** Need for explainable predictions in high-stakes decisions
- **Human Oversight:** Automated systems should assist, not replace human judgment

### Best Practices Implemented
✅ Multiple evaluation metrics (not just accuracy)  
✅ Statistical validation of all findings  
✅ Interpretable models (feature importance analysis)  
✅ Explicit documentation of limitations  
✅ Emphasis on human-AI collaboration  

### 🎯 Recommendation

These models should be used as **decision-support tools**, not as autonomous fact-checkers. Human oversight remains essential for responsible deployment in misinformation detection systems.  

---

## 🚀 Future Work

### Technical Enhancements
- **Transformer Models:** Fine-tune BERT, RoBERTa for improved performance
- **Multimodal Analysis:** Incorporate image and video analysis
- **Feature Expansion:** Add source credibility, user engagement metrics
- **Cross-Lingual Detection:** Extend to multiple languages

### Advanced Techniques
- **Ensemble Methods:** Combine multiple models for better accuracy
- **Active Learning:** Efficient labeling of new data
- **Adversarial Training:** Improve robustness against evasion
- **Explainable AI:** Implement LIME/SHAP for better interpretability

### Real-World Deployment
- **API Development:** Real-time fake news detection service
- **Browser Extension:** On-the-fly fact-checking while browsing
- **Mobile Application:** Portable misinformation verification tool
- **Continuous Learning:** Automated pipeline for model updates

### Research Directions
- **Domain Adaptation:** Transfer learning across news categories
- **Claim Verification:** Move from article-level to claim-level detection
- **Propagation Analysis:** Study how misinformation spreads on social networks
- **Bias Mitigation:** Fairness-aware model training

---

## 📄 References

### Key Papers

1. Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). FakeNewsNet: A data repository with news content, social context and dynamic information. *arXiv:1809.01286*.

2. Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING 2018*.

3. Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. *ACM Computing Surveys, 53*(5), 1-40.

4. Kaliyar, R. K., Goswami, A., & Narang, P. (2021). FakeBERT: Fake news detection with BERT-based deep learning. *Multimedia Tools and Applications, 80*, 11765-11788.

5. Lazer, D. M., et al. (2018). The science of fake news. *Science, 359*(6380), 1094-1096.

**Complete bibliography available in the notebook.**

---

## 🛠️ Tools & Technologies

**Core Stack:**
- Python 3.8+
- pandas, NumPy (data manipulation)
- scikit-learn (machine learning)
- NLTK (text processing)
- matplotlib, seaborn (visualization)
- scipy (statistical testing)
- Jupyter Notebook (reproducible analysis)

**Key Techniques:**
- TF-IDF vectorization
- Stratified train-test splitting
- Cross-validation
- Statistical hypothesis testing
- Feature importance analysis

---

## 🤝 Contributing

This is an academic project for portfolio demonstration. If you have suggestions or find issues, feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact the author via GitHub

---

## 📧 Contact

**Sanjay Kumar Chhetri**  
- GitHub: [@sanjaykshetri](https://github.com/sanjaykshetri)
- Repository: [Misinformation-Detection-ML-Model2](https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2)

---

## 📜 License

This project is open source and available for educational and research purposes.

**Dataset Credit:**  
FakeNewsNet dataset by Kai Shu et al. - [Original Repository](https://github.com/KaiDMML/FakeNewsNet)

---

## 🙏 Acknowledgments

- **FakeNewsNet Team** for providing the comprehensive dataset
- **PolitiFact** and **GossipCop** for fact-checking labels
- **Research Community** for establishing methodological foundations
- **Open Source Libraries:** scikit-learn, pandas, NumPy, matplotlib, seaborn

---

## 📊 Project Status

✅ **Completed:** Data analysis, model training, evaluation  
✅ **Documented:** Comprehensive notebook with literature context  
✅ **Validated:** Statistical rigor and reproducible results  
🔄 **Ongoing:** Exploring advanced techniques and deployment options

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

---

*Last Updated: December 16, 2025*