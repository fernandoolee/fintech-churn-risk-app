# Fintech New-User Churn Risk (Streamlit app)

Upload a CSV of new digital-wallet users and get, for each user, the **probability of uninstalling within 30 days** and a **risk tier** (Low / Medium / High / Very high) with suggested CRM actions.

The method comes from my IEEE paper:
> F. Lee and N. Zulkarnain, "Predicting New User Uninstallation in Digital Wallet Applications with Explainable Artificial Intelligence," *ICIMTech 2026*. [IEEE Xplore](https://ieeexplore.ieee.org/document/11698998) · research code: [churn-prediction-xai-digital-wallet](https://github.com/fernandoolee/churn-prediction-xai-digital-wallet)

> **Data note:** the model in this repo is trained on **synthetic data** with the same schema as the research dataset. It demonstrates the workflow and does not reflect any real company's users or figures.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use sample data** to try it without your own file.

## Files
| File | What it does |
|---|---|
| `app.py` | Streamlit UI: upload → score → risk tiers → download |
| `features.py` | Preprocessing shared by training and the app (cleaning, encoding, engagement score, total amount, weekday) |
| `train_model.py` | Retrains the model on a synthetic CSV and writes `model.json` + `model_meta.json` |
| `model.json` | XGBoost model in native JSON format (no pickle) |
| `model_meta.json` | Feature order, risk-tier cut-offs, test ROC-AUC |
| `sample_new_users.csv` | 1,000 synthetic users to try the app |

## How the tiers work
Users are ranked by predicted probability. Cut-offs are set on the synthetic test set so that roughly the bottom 40% are **Low**, the next 30% **Medium**, the next 20% **High** and the top 10% **Very high**.

Author: **Fernando Lee** · [Portfolio](https://fernandoolee.github.io/Portofolio-Fernando-Lee/) · [LinkedIn](https://www.linkedin.com/in/fernando-lee-67835032a)
