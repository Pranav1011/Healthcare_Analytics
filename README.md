# 🏥 Healthcare Analytics Dashboard

A **Streamlit-based interactive dashboard** designed to analyze hospital data, optimize resource allocation, and predict patient wait times using machine learning.

---

## 🚀 Features

1. **Visual Analytics**:
   - Interactive charts for hourly admissions, department utilization, and transfer patterns.
   - Department-specific metrics like average length of stay.

2. **Predictive Modeling**:
   - Machine learning model to predict emergency department (ED) wait times.
   - Features a trained Random Forest and XGBoost model.

3. **Easy Deployment**:
   - Fully Dockerized for consistent deployment.
   - Supports local and cloud deployment.

---

## 🛠️ Setup Instructions

### **1. Prerequisites**
- Python 3.9+
- Docker installed on your system

### **2. Run Locally**

#### **Step 1: Clone the Repository**

git clone https://github.com/<your-username>/healthcare-analytics.git
cd healthcare-analytics

#### **Step 2: Create and Activate a Virtual Environment**

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

#### **Step 3: Install Dependencies**
pip install -r requirements.txt

#### **Step 4: Run the Streamlit App**
streamlit run streamlit_app.py

Open your browser at http://localhost:8501.

## **3. Run with Docker**

#### **Step 1: Build the Docker Image**
docker build -t healthcare-analytics .

#### **Step 2: Run the Docker Container**
docker run -p 8501:8501 healthcare-analytics


🧠 Predictive Modeling

Trained Model
	•	Random Forest and XGBoost models trained on ED wait time prediction.
	•	Data preprocessing includes one-hot encoding and feature engineering.


📊 Visualizations
	1.	Hourly Admission Patterns:
	•	Interactive bar chart showing admissions by type and time.
	2.	Department Utilization:
	•	Average and median length of stay for each ward.
	3.	Transfer Patterns:
	•	Hourly transfer data visualized with a multi-colored bar chart.

    
🏗️ Deployment Options
	1.	Docker: Fully containerized for easy deployment on any system.
	2.	Cloud Deployment (Optional):
	•	AWS Elastic Beanstalk
	•	Google Cloud Run
	•	Heroku

📝 Notes
	•	Data is sourced from the MIMIC-III Clinical Database.
	•	This project is for educational and research purposes only.
