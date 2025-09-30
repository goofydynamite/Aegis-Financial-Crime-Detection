Project Aegis: A Machine Learning Approach to Financial Crime Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)


## Project Goal
The goal of Project Aegis is to develop a proactive, data-driven system for identifying and analyzing sophisticated financial crime networks. This project moves beyond simple, rule-based alerts to a machine learning model that can detect anomalous patterns in a large volume of transaction data, ultimately providing actionable intelligence to investigators through an interactive dashboard.

##  The Problem
Financial institutions face the immense challenge of combating crimes like money laundering, which involve complex networks of transactions designed to obscure the source of illicit funds. Traditional methods often rely on static rules that are easily circumvented by sophisticated criminals. This creates a need for intelligent systems that can learn normal behavior and flag suspicious deviations at a network level, reducing false positives and allowing investigators to focus on the highest-priority threats.

---

##  Key Features
- **Realistic Data Simulation**: A robust Python script generates a dataset of over 20,000 transactions across 1,000 accounts, with engineered crime patterns including **Layering**, **Smurfing (Structuring)**, and **Rapid Pass-Through** schemes.
- **Rich Feature Engineering**: The system enriches raw data with crucial risk indicators such as transaction velocity, country risk flags (`is_foreign_transaction`), and transaction types.
- **Unsupervised Anomaly Detection**: An **Isolation Forest** model is trained to identify outliers. This unsupervised approach is ideal as it doesn't require pre-labeled fraud data and can adapt to new, unseen criminal patterns.
- **Performance Validation**: The model's effectiveness is rigorously measured using a confusion matrix (True Positives, False Positives), and a **Model Precision** score is calculated to quantify the reliability of its alerts.
- **Interactive SOC Dashboard**: A multi-page Power BI report serves as a Security Operations Center (SOC) dashboard, allowing analysts to move from a high-level overview to a deep-dive network graph investigation of specific alerts.

---

## 📊 Final Dashboard
The final output is a comprehensive, multi-page Power BI dashboard designed for financial crime investigators.



---

## ⚙️ System Architecture
The project follows a standard analytical pipeline, moving from data generation to final visualization.
Technical Deep Dive: The Methodology



1.  **Data Generation**: The foundation of the project is a simulated dataset created with Python's `Faker` library. It includes realistic features like account creation dates, transaction types, and merchant details. Crucially, specific crime typologies are programmatically injected:

    - **Smurfing**: One account makes numerous small deposits into many other accounts.

    - **Layering**: Funds are passed through a circular chain of accounts in rapid succession.

    - **Pass-Through**: A large sum is deposited from a high-risk source and immediately transferred out.



2.  **Feature Engineering**: Raw data is insufficient for effective modeling. New features were created in Power Query and Python to provide context:

    - `Crime_Type`: A categorical label derived from the known `transaction_id`s of engineered patterns.

    - `Is_Actual_Crime`: A binary "ground truth" column (1/0) used for model validation.

    - *(In a full implementation, features like `account_age`, `avg_transaction_value`, `transaction_frequency` would be created here).*



3.  **Machine Learning Model**: An **Isolation Forest** was chosen for anomaly detection.

    - **Why?** It's an unsupervised model, which is perfect for this problem as we want to find *anomalies* of all kinds, not just the patterns we already know. It works by "isolating" outliers in the data, making it computationally efficient and effective.

    - The model assigns an `anomaly_score` to each transaction. The lower the score, the more anomalous the transaction.


4.  **Model Performance**: To validate the model, DAX measures were created in Power BI to build a confusion matrix based on a defined risk threshold (e.g., `anomaly_score < 0.1`).


| Metric          | Value        | Description                                                 |
| --------------- | ------------ | ----------------------------------------------------------- |
| **True Positives** | *[Your TP #]* | The model correctly identified a known criminal transaction.    |
| **False Positives** | *[Your FP #]* | The model incorrectly flagged a normal transaction.         |
| **Model Precision** | *[Your %]* | Of all the alerts generated, this % were actual crimes.     |


Metric	Value	Description
True Positives	48	The model correctly identified a known criminal transaction.
False Positives	112	The model incorrectly flagged a normal transaction.
Model Precision	30.0%	Of all the alerts generated, this % were actual crimes.
Tech Stack
- **Data Analysis & Modeling**: Python (Pandas, Scikit-learn, Faker, TA)
- **Data Transformation**: Power Query (M Language)
- **Dashboarding & DAX Measures**: Microsoft Power BI
- **Development Environment**: VS Code, Jupyter Notebook







📂 Repository Structure
Aegis-Financial-Crime-Detection/
│
├── data/
│   ├── raw/
│   │   └── simulated_transactions.csv
│   └── processed/
│       └── enriched_transactions_for_dashboard.csv
│
├── dashboards/
│   ├── Aegis_Dashboard.pbix
│   └── dashboard_screenshot.png
│
├── notebooks/
│   ├── 01_Data_Simulation.ipynb
│   └── 02_Analysis_and_Modeling.ipynb
│
├── src/
│   ├── init.py
│   ├── data_processing.py
│   └── modeling.py
│
├── .gitignore
├── README.md
└── requirements.txt

# 🔧 Setup & Usage

To set up and run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Aegis-Financial-Crime-Detection.git](https://github.com/your-username/Aegis-Financial-Crime-Detection.git)
    cd Aegis-Financial-Crime-Detection
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
s   pip install -r requirements.txt
    ```
4.  **Run the analysis:** Open the Jupyter Notebooks in the `notebooks/` directory to explore the data simulation and modeling process.
5.  **View the dashboard:** Open the `Aegis_Dashboard.pbix` file in the `dashboards/` directory using Power BI Desktop.

---

## 🔮 Future Scope
- **Graph Database Integration**: Migrate the data to a graph database like Neo4j to run powerful graph algorithms (e.g., PageRank, community detection) to find influential launderers and hidden criminal rings.
- **Real-Time Alerting**: Deploy the model via a FastAPI endpoint and connect it to a streaming platform like Kafka to score transactions in real-time.
- **Explanatory AI (LLM)**: Integrate an LLM to automatically generate a human-readable "dossier" for each high-risk alert, explaining in natural language why the transaction was flagged based on its features.
