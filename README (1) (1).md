#  Medical Insurance Cost Prediction

A machine learning project that predicts an individual's **medical insurance charges** using
**Multiple Linear Regression**, based on their age, BMI, number of children, smoking status, sex,
and region.

## Problem Statement

Health insurers need to estimate a customer's expected medical costs before setting a premium.
Pricing every customer the same ignores real risk differences — smokers, older customers, and
higher-BMI customers tend to cost significantly more. This project builds a regression model that
estimates annual insurance charges from a customer's profile, so premiums can be set based on
predicted risk rather than a flat rate.

## Dataset

**Medical Cost Personal Dataset** (`insurance.csv`) — 1,338 records, 7 columns:

| Column | Description |
|---|---|
| `age` | Age of primary beneficiary |
| `sex` | Female / male |
| `bmi` | Body mass index |
| `children` | Number of dependents covered |
| `smoker` | Smoking status (yes/no) |
| `region` | US residential region (NE/NW/SE/SW) |
| `charges` | **Target** — annual medical insurance cost billed (USD) |

## Project Structure

```
.
├── insurance.csv                 # Raw dataset
├── insurance_prediction.ipynb    # Phases 1-6: EDA, cleaning, training, evaluation
├── insurance_model.pkl           # Trained model + metrics (produced by the notebook)
├── app.py                        # Phase 7: Streamlit prediction app
├── requirements.txt
└── README.md
```

## Workflow

1. **Data Cleaning & Preprocessing** — checked for missing values and duplicates (none found),
   then encoded `sex` and `smoker` as binary and one-hot encoded `region`.
2. **Exploratory Data Analysis** — visualized charges vs. age, BMI, and smoking status, and
   examined a correlation heatmap. `smoker` is by far the strongest predictor of cost, followed by
   `age` and `bmi`.
3. **Model Training** — an 80/20 train/test split, then a `LinearRegression` model fit on 8 features
   (`age`, `sex`, `bmi`, `children`, `smoker`, and 3 one-hot region columns).
4. **Evaluation** — measured on the held-out test set.

## Model Performance

| Metric | Value |
|---|---|
| MAE | ≈ $4,177 |
| MSE | ≈ 35,478,021 |
| RMSE | ≈ $5,956 |
| **R² Score** | **≈ 0.807** |

The model explains about 81% of the variance in insurance charges. It captures the main cost
drivers well (especially the smoker/non-smoker split) but, being a single straight-line model, it
doesn't fully capture the *interaction* between smoking, age, and BMI — an older, high-BMI smoker
costs disproportionately more than each factor would suggest on its own.

## Running the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-run the full analysis and retrain the model
jupyter notebook insurance_prediction.ipynb

# 3. Launch the prediction app
streamlit run app.py
```

## Prediction App (Phase 7)

The Streamlit app takes age, sex, BMI, number of children, smoking status, and region as input and
returns an estimated annual insurance cost using the trained model.

**Example test cases:**
| Profile | Predicted Cost |
|---|---|
| 25-year-old female, non-smoker, BMI 22, no children, Northeast | ≈ $2,124 |
| 52-year-old male, smoker, BMI 34.5, 2 children, Southeast | ≈ $36,013 |

## Technologies Used

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

## Limitations

- Trained on a single, relatively small (1,338-row) public dataset — may not generalize to other
  populations or regions.
- Linear Regression assumes additive, linear relationships and cannot capture interaction effects
  (e.g. smoking × BMI × age) without manual feature engineering.
- Does not account for pre-existing conditions, occupation, or other real-world underwriting
  factors an actual insurer would use.

## Author

**Shahla Munawar**
- GitHub: [github.com/shahla2006](https://github.com/shahla2006)
- Repository: [github.com/shahla2006/Medical-Insurance-Cost-Prediction](https://github.com/shahla2006/Medical-Insurance-Cost-Prediction)
