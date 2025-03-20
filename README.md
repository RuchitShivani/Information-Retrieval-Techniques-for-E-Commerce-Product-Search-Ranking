# E-commerce Product Search & Ranking

## Overview
This project aims to enhance the efficiency of e-commerce product search by implementing **Information Retrieval (IR) techniques** and **Machine Learning-based ranking algorithms**. Our solution improves search relevance by leveraging advanced ranking methodologies such as **BM25, Dense Embeddings, and BERT-based Ranking**.

## Features
- **Query Understanding**: Uses NLP techniques to analyze and interpret user search queries.
- **Candidate Retrieval**: Efficiently fetches relevant product listings using traditional IR models and neural embeddings.
- **Re-ranking**: Applies deep learning models to refine and improve the ranking of retrieved products.
- **Personalization**: Adjusts ranking scores based on user behavior, past interactions, and purchase history.
- **Scalability**: Optimized to handle large-scale product datasets with minimal latency.

## Dataset
The project utilizes the **Amazon Product Dataset**, which includes:
- Product descriptions
- Customer reviews and ratings
- Metadata such as category, price, and brand

### Data Preprocessing
- Tokenization and text normalization
- Stopword removal
- Embedding-based feature extraction
- Handling missing or inconsistent data

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ecommerce-search-ranking.git
cd ecommerce-search-ranking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Data Preprocessing
```bash
python preprocess.py
```

### 4. Train the Model
```bash
python train.py
```

### 5. Evaluate the Model
```bash
python evaluate.py
```

## Architecture
The system follows a **two-stage retrieval and ranking pipeline**:

### **1. Candidate Retrieval**
This stage quickly fetches a set of relevant product candidates from a large dataset using:
- **BM25**: A traditional term-weighting approach.
- **Dense Embeddings**: Pre-trained embeddings (e.g., **SBERT**) for improved semantic understanding.
- **Approximate Nearest Neighbors (ANN)**: Efficiently searches for the closest vector representations of query-related products.

### **2. Re-ranking**
The retrieved candidates are re-ranked using:
- **BERT-based ranking models** for fine-grained ranking adjustments.
- **Click-through rate (CTR) prediction models** for personalized recommendations.
- **Hybrid techniques combining lexical and neural ranking methods.**

## Model Performance

### Evaluation Metrics
We assess our ranking models using:
- **Mean Reciprocal Rank (MRR)**
- **Normalized Discounted Cumulative Gain (NDCG)**
- **Click-Through Rate (CTR) improvement**

### Results
- Achieved **XX%** improvement in Mean Reciprocal Rank (MRR).
- Increased **Click-Through Rate (CTR)** by **XX%**.
- Outperformed baseline models such as **BM25 and TF-IDF**.

## Fraud Detection Mechanism
Our system incorporates **fraud detection techniques** to prevent misleading product rankings. The methods include:
- **Anomaly Detection Models**: Identifies unusual activity in product reviews and seller ratings.
- **Graph-Based Detection**: Analyzes seller-product relationships to flag potential fraudulent sellers.
- **Behavioral Analysis**: Tracks user interactions for suspicious patterns.
- **Natural Language Processing (NLP) for Fake Reviews**: Detects unnatural patterns in product reviews using deep learning models.

## Deployment

### API Endpoints
We provide a **REST API** for integrating the search and ranking system into e-commerce platforms. Key endpoints include:
- `/search?query=product_name` – Returns ranked product results.
- `/recommendations?user_id=XYZ` – Personalized recommendations based on user history.
- `/fraud-detection` – Runs fraud analysis on product listings.

### Deployment Steps
- **Model Training**: Preprocess data, train ranking models, and evaluate results.
- **Dockerization**: Package the API into a **Docker container** for easy deployment.
- **Cloud Hosting**: Deploy on **AWS/GCP** with auto-scaling capabilities.

## References

1. **R. Nogueira, W. Yang, J. Lin** – "Multi-stage document ranking with BERT", 2020 IEEE Conference on Information Retrieval and NLP.
2. **Z. Zhao, P. Zheng, S. Xu** – "Object detection with deep learning: A review", IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 11, pp. 3212-3232, 2019.
3. **A. Vaswani et al.** – "Attention is all you need", NeurIPS, 2017.
4. **W. Jiang and L. Zhang** – "Graph-based fraud detection in e-commerce platforms", Journal of Machine Learning Research, 2022.
5. **S. Robertson and H. Zaragoza** – "The Probabilistic Relevance Framework: BM25 and Beyond", Information Retrieval, 2009.

Here are attached screenshots.

![earphones2](https://github.com/user-attachments/assets/c97c0d36-42da-414f-8e55-aa26ae506aa3)
![gaming2](https://github.com/user-attachments/assets/a5b89703-9263-4e47-84d1-938f8371c5fd)
![gaming4](https://github.com/user-attachments/assets/42d15dfd-8c75-4ab5-8101-548d4911143a)
![earphones3](https://github.com/user-attachments/assets/5019bd25-a02c-476b-8f20-7db3fe9d3de6)
![earphones4](https://github.com/user-attachments/assets/8f962013-9404-4ac9-93ee-e4a2da93f8d5)
![earphones1](https://github.com/user-attachments/assets/d75fdf1c-dd45-4c14-b27e-084a731098ba)
![gaming3](https://github.com/user-attachments/assets/8b9df846-2979-4b16-ac7e-7af934e758c4)
![gaming1](https://github.com/user-attachments/assets/a8e7d45c-a952-4a85-ba2c-ac9e05c1691f)


## Screenshots![gaming5](https://github.com/user-attachments/assets/278f4494-24fd-4292-9810-b8aba49f567e)



