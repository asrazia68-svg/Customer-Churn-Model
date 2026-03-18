# 🚀 Customer Churn Prediction for E-Commerce

## 🎯 Project Overview
Predict which customers are likely to **churn** in an e-commerce platform using a **Gradient Boosting Classifier** pipeline.  
This project handles:
- Data preprocessing  
- Feature engineering  
- Imbalanced class handling with **SMOTE**  
- Model evaluation & visualization  

The goal is to **identify at-risk customers** and help the business retain them proactively.

---

## 🗂 Dataset
| Feature | Type | Description |
|---------|------|-------------|
| Customer_ID | Identifier | Unique customer ID (dropped during modeling) |
| account_age_months | Numeric | How long the customer has been registered |
| avg_order_value | Numeric | Average order value per customer |
| total_orders | Numeric | Total orders placed |
| days_since_last_purchase | Numeric | Used to create the `Churn` target |
| discount_usage_rate | Numeric | Percentage of discounts used |
| return_rate | Numeric | Product return rate |
| customer_support_tickets | Numeric | Number of tickets submitted |
| loyalty_member | Categorical | Membership type/status |
| browsing_frequency_per_week | Numeric | Average website visits per week |
| cart_abandonment_rate | Numeric | % of abandoned carts |
| product_review_score_avg | Numeric | Average review score given |
| engagement_score | Numeric | Activity-based engagement metric |
| satisfaction_score | Numeric | Dropped during modeling |
| price_sensitivity_index | Numeric | Sensitivity to price changes |

**Target column:** 
- `Churn` → Created from `days_since_last_purchase > 30`  
  - 1 = Churned customer  
  - 0 = Non-Churned customer

---

## 🛠 Data Preprocessing
1. **Dropped leakage features:** `Customer_ID`, `satisfaction_score`, `days_since_last_purchase`  
2. **Categorical encoding:** `loyalty_member` using One-Hot Encoding  
3. **Standardized numeric features** using `StandardScaler`  
4. **Handled class imbalance** with **SMOTE** oversampling  

---

## 🏗 Modeling
**Algorithm:** Gradient Boosting Classifier  
**Key Parameters:**
- `n_estimators=150`  
- `learning_rate=0.01` (slow learning to prevent overfitting)  
- `max_depth=2` (shallow trees for generalization)  
- `subsample=0.8` (stochastic boosting)  
- `max_features='sqrt'`  

**Pipeline Steps:**
1. Column transformations (numeric + categorical)  
2. SMOTE oversampling  
3. Gradient Boosting classifier  

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy | 0.88 |
| Test Accuracy | 0.867 |
| Churn Recall | 0.84 |
| Non-Churn Precision | 0.91 |

---

## 📈 Evaluation Visualizations

1. **Confusion Matrix**  
2. **Correlation Heatmap**  
3. **Churn Distribution**  
4. **Boxplots for Avg Order Value & Total Orders by Churn**  
5. **Feature Importance**

These visualizations help understand which features most influence customer churn.

---

## ⚡ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn.git


