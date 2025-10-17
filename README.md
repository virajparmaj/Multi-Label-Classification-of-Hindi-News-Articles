# Multi-Label Classification of Hindi News Articles
_A machine learning system for empowering local Hindi news organizations_

---

## 📌 Overview
### Problem Statement
Local news organizations in India, especially those serving vernacular audiences, face budget constraints that limit their ability to adopt modern data tools. With ~98% of rural and 1/3 of urban Indians preferring Hindi content, bridging the Hindi–English digital divide is critical.

This project builds a **multi-label text classifier** for Hindi e-news articles, enabling automated tagging into **14 categories**. The system empowers newsrooms, editors, and readers with faster curation, improved accessibility, and cost-effective AI adoption.

### Motivation
- Support regional journalism and preserve linguistic diversity.  
- Provide vernacular readers with more accurate, timely, and categorized content.  
- Enable targeted advertising and insights into content trends.  

---

## ✨ Features
- **Custom Hindi dataset** scraped from online news portals.  
- **Multi-label classification** across 14 categories.  
- **TF-IDF feature extraction** with scikit-learn pipelines.  
- **Binary Relevance baseline** achieving ~43% accuracy.  
- **Lightweight and reproducible codebase** for academic and industrial extension.  

### Architecture Diagram
*(Placeholder – insert an image of the pipeline here)*  

---

## ⚙️ Installation and Setup
Clone the repository and install dependencies:

```bash
# Clone repo
git clone https://github.com/virajparmaj/Multi-Label-Classification-of-Hindi-News-Articles.git
cd Multi-Label-Classification-of-Hindi-News-Articles

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
````

## 📂 Datasets, Models, and Tools

* **Dataset**: Custom-scraped Hindi e-news dataset (14 labels).
* **Preprocessing**: NLTK / iNLTK for tokenization and stopword removal.
* **Feature Extraction**: TF-IDF.
* **Model**: Binary Relevance baseline with scikit-learn classifiers.
* **Language**: Python.
* **Libraries**: scikit-learn, pandas, numpy.

---

## 📊 Results

* **Baseline Accuracy (Binary Relevance + TF-IDF)**: ~43%
* **Evaluation Split**: 80/20 train-test.
* **Metrics**: Precision, Recall, F1-score recorded per label.

---

## 👥 Contributors & Acknowledgments

* **Viraj Parmaj** – Dataset creation, model implementation, preprocessing support.
* **Sanu Raj** – Model implementation, evaluation
* **Prof. Dr. Pratistha Mathur (Supervisor)** – Guidance at Manipal University Jaipur.

Special thanks to the open-source community for scikit-learn and iNLTK.

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
