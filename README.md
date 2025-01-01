# Telco Customer Churn
Telco Customer Churn Prediction Using Machine Learning


## Project Overview
This project aims to predict customer churn in a telecommunications company by using machine learning models. Customer churn refers to the rate where customers stop doing business with an entity. By identifying the factors that drive churn and predicting the probability of customer churn, companies can implement more effective strategies to maintain customers and reduce revenue loss.


## Business Problem
Customer churn is a critical issue for the telecommunications industry, which is directly impacting revenue and business growth. Understanding the reasons behind churn and predicting of which customers are most likely to leave can help companies take more proactive action to improve customer retention.


## Objective
The main objective of this project is to develop a predictive model to identify customers who are risky of switching to other companies and to discover the most significant causes of customer churn. This will allow companies to implement interventions targeted to reducing churn.


## Data
The dataset used in this project includes customer information such as demographic details, service usage, account information, and churn status. Key features include:
- **Demographics:** Gender, senior status, spouse, and dependents.
- **Account Information:** Contract type, payment method, monthly fee and total fee.
- **Service Usage:** Type of internet service, online security, device protection and more.
- **Churn Status:** Customers who left within the last month (This column is called Churn).


## Steps
1. **Level 1: Understanding the Terrain:** Analyze the dataset to understand customer behavior and churn patterns.
2. **Level 2: Finding the Drivers of Churn:** Identify the most important features influencing churn.
3. **Level 3: Building a Predictive Model:** Develop machine learning models to predict churn, evaluate the performance of the models, and interpret the results from the best-performing model.
4. **Level 4: Presenting the Findings:** Provide actionable recommendations based on insights.


## Early Insight From The Data
- Feature tenure (subscription duration) is the numerical feature with the most significant impact on churn with a negative correlation. Here is the Correlation table between churn and other numeric columns.

    | Numeric Feature   | Correlation with Churn    |
    |-------------------|---------------------------|
    | Tenure            | -0.352229                 |
    | MonthlyCharges    | 0.193356                  |
    | TotalCharges      | -0.199484                 |

- The categorical features that affect churn the most are additional service types (OnlineSecurity, TechSupport, etc.), type of contract, and payment method.

    ![Churn Distribution by Categorical Feature](/images/churn-distribution-by-categorical-feature.png)


- Feature Contract is the feature that has the most significant effect on churn based on the results of the Chi-square test.

    | Feature Name      | Chi2      | p-value        | Interpretation                                                       |
    |-------------------|------------|---------------|----------------------------------------------------------------------|
    | Gender            | 0.48       | 0.48658       | Gender does not have a significant relationship with churn.          |
    | Partner           | 158.73     | 0.00000       | Partner has a significant relationship with churn.                   |
    | Dependents        | 189.13     | 0.00000       | Dependents have a significant relationship with churn.               |
    | PhoneService      | 0.92       | 0.33878       | PhoneService does not have a significant relationship with churn.    |
    | MultipleLines     | 11.33      | 0.00346       | MultipleLines has a significant relationship with churn.             |
    | InternetService   | 732.31     | 0.00000       | InternetService has a significant relationship with churn.           |
    | OnlineSecurity    | 850.00     | 0.00000       | OnlineSecurity has a significant relationship with churn.            |
    | OnlineBackup      | 601.81     | 0.00000       | OnlineBackup has a significant relationship with churn.              |
    | DeviceProtection  | 558.42     | 0.00000       | DeviceProtection has a significant relationship with churn.          |
    | TechSupport       | 828.20     | 0.00000       | TechSupport has a significant relationship with churn.               |
    | StreamingTV       | 374.20     | 0.00000       | StreamingTV has a significant relationship with churn.               |
    | StreamingMovies   | 375.66     | 0.00000       | StreamingMovies has a significant relationship with churn.           |
    | Contract          | 1184.60    | 0.00000       | Contract has a significant relationship with churn.                  |
    | PaperlessBilling  | 258.28     | 0.00000       | PaperlessBilling has a significant relationship with churn.          |
    | PaymentMethod     | 648.14     | 0.00000       | PaymentMethod has a significant relationship with churn.             |
    | SeniorCitizen     | 159.43     | 0.00000       | SeniorCitizen has a significant relationship with churn.             |



## Data Preparation
Data processing steps include:
1. **Handling Missing Values:** Imputing missing values for features that have missing values, such as TotalCharges feature.
2. **Feature Selection:** Removing features that are considered less impact on churn from the analysis results to improve model performance.
3. **Feature Encoding:** Converting categorical features into numerical representations. This feature encoding was performed using two methods, namely one-hot encoding and label encoding.
4. **Handling Class Imbalance:** SMOTE was used for oversampling the minority class (churn) to balance the dataset.
5. **Normalization:** Min-max normalization was applied to numeric features for consistent scaling.
6. **Data Split:** The data was divided into 70% for training and 30% for testing and validating the performance of the model.


## Machine Learning Modeling
This project will use four different machine learning algorithms, as shown below.
1. **Support Vector Machine (SVM):** This algorithm is suitable for non-linear data and produces strong decision margins.
2. **Logistic Regression:** A simple algorithm, easy to interpret, and suitable for binary classification cases.
3. **Random Forest:** This ensemble-based model algorithm can handle non-linear data well and provide information on the importance of features.
4. **XGBoost:** This algorithm is pretty effective for high performance classification as it combines weaker models.


## Model Evaluation
The classification report of the model that has been trained using test data is as shown below.
| Model | Accuracy | Precision | Recall | F1-Score |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Support Vector Machine    | 0.84 | 0.84 | 0.84 | 0.84 |
| Logistic Regression       | 0.81 | 0.81 | 0.81 | 0.81 |
| Random Forest             | 0.86 | 0.86 | 0.86 | 0.86 |
| XGBoost                   | 0.85 | 0.85 | 0.85 | 0.85 |


Besides the classification report results, the evaluation of the model also used metrics AUC-ROC. The results of the evaluation of the AUC-ROC metrics are shown below.

![Hraph of AUC-ROC Metrics](/images/graph-of-auc-roc-metrics.png)

Based on the results of the evaluation both of the classification report, and the ROC-AUC, the model using the XGBoost algorithm is the best model. This is proven by the following evidence.
- The XGBoost model has the highest AUC score among all models which is 0.94.
- This model produces very balanced precision and recall.
- The overall accuracy of this model is pretty high at 85%.
- The F1-score generated from this model also consistently shows advantage.


## Feature Importance
The visualization of the feature importance of the best model, the XGBoost model, is shown below.

![Feature Importance from XGBoost](/images/feature-importance-from-xgboost.png)

Based on the feature importance visualization graph, the Contract_Month-to-month feature is the most influential feature in the model. Followed by the PaymentMethod_Electronic check feature.


## Business Recommendation
Based on the results of the entire process, here are the Business Recommendations to reduce customer churn and increase retention.

- Increase Retention Through Long-Term Contracts, the company needs to encourage customers to switch from monthly contracts to annual or biennial contracts through incentives such as discounts, service bonuses, or cashback.

- Focus on New Customers with Short Subscription Periods, the company needs to introduce loyalty programs designed specifically for new customers, such as a free month bonus after 6 months or a discount on the first year of subscription.

- Promote Stable Payment Methods and Minimize the Risk of Electronic Checks, the company needs to promote automatic payment methods such as Credit Card (automatic) or Bank Transfer (automatic) with incentives such as bill discounts or small gifts. In addition, the company also needs to expand education and promotion to reduce the use of Electronic Check method, which is strongly associated with churn.

- Improve and Promote Additional Services (OnlineSecurity & TechSupport), the company needs to make services like OnlineSecurity and TechSupport as part of the main package or offer affordable upgrades. The company also needs to promote these additional services as important features to improve customer experience.

- Set Prices and Service Offerings Wisely, the company should segment customers based on the level of monthly charges (MonthlyCharges) and total bills (TotalCharges) to offer packages that suit their abilities.
The company should also consider providing discount packages or bundling services for customers with high bills to increase satisfaction.


## Conclusion
The project successfully identified the most significant factors driving customer churn and built a predictive model with high accuracy. The insights and business recommendations provided can help businesses reduce churn rates and improve customer retention strategies.


