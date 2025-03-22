
# 📱 Mobile Price Prediction Web App  

Welcome to the **Mobile Price Prediction** project! This is an end-to-end machine learning application designed to predict mobile phone prices based on their specifications. Built using **Streamlit** and deployed on **Hugging Face**, this web app provides users with an intuitive interface for estimating mobile phone prices using a regression model.  

## 🚀 **Overview**  
- Developed a **Mobile Price Prediction** web app using a complete **Machine Learning Pipeline**.  
- Utilized **Optuna** for hyperparameter tuning to select the best model.  
- Applied **StandardScaler** for feature scaling.  
- Built multiple regression models and selected the most accurate one.  
- Deployed the app using **Streamlit** on **Hugging Face**.  

## 🌟 **Features**  
- Predict mobile phone prices based on key specifications.  
- Compare predictions using different regression models.  
- User-friendly and interactive interface.  
- Real-time price predictions.  

## 🛠️ **Tech Stack**  
- **Python**  
- **Pandas, NumPy** for data manipulation  
- **Matplotlib, Seaborn** for visualization  
- **Scikit-Learn** for machine learning models  
- **Optuna** for hyperparameter tuning  
- **Streamlit** for building the web app  
- **Hugging Face** for deployment  

---

## 📊 **Data Description**  
The dataset contains various mobile phone specifications and their respective prices.  
Key Features include:  
- **Battery Power (mAh)**  
- **RAM (GB)**  
- **Storage (GB)**  
- **Processor Speed (GHz)**  
- **Camera Resolution (MP)**  
- **Screen Size (Inches)**  
- **Price (Target Variable)**  

---

## 🔎 **Approach**  

1. **Data Exploration**  
   - Conducted exploratory data analysis (EDA) to identify patterns and correlations.  

2. **Data Preprocessing**  
   - Applied **StandardScaler** to normalize the data for optimal model performance.  

3. **Model Building**  
   - Built and evaluated regression models using **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting**.  

4. **Hyperparameter Tuning**  
   - Applied **Optuna** for efficient hyperparameter tuning to select the best model.  

5. **ML Pipeline Creation**  
   - Created a machine learning pipeline integrating data preprocessing, model training, and prediction.  

6. **Model Serialization**  
   - Saved the optimized model using **Pickle**.  

7. **Deployment**  
   - Developed the app using **Streamlit** and deployed it on **Hugging Face**.  

---

## 🧑‍💻 **How to Run the Project**  

1. Clone the repository:  
    ```bash
    git clone https://github.com/Saiphani2105/Mobile_Price_Prediction.git
    cd mobile-price-prediction
    ```

2. Install the required dependencies:  
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:  
    ```bash
    streamlit run app.py
    ```

4. Access the app at `http://localhost:8501`  

---

## 🚀 **Live Demo**  
Try out the app here:  
[Hugging Face Deployment](https://huggingface.co/spaces/Phaneendrabayi/Mobile_price_prediction)

---

## 📁 **Project Structure**  
```bash
mobile-price-prediction/
│
├── data/
│   ├── mobile_data.csv
│
├── models/
│   ├── best_model.pkl
│
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
```

- `data/` - Contains the dataset.  
- `models/` - Stores the best-trained model using Pickle.  
- `app.py` - Streamlit web app for price prediction.  
- `model_training.py` - Code for data preprocessing, model training, hyperparameter tuning, and model selection.  
- `requirements.txt` - List of dependencies.  
- `README.md` - Project documentation.  

