# Kaggle Playground Series 6

All my ongoing submission for all 12 episodes (months) of Kaggle playground series 6 (2026)

---

### January ☃️

**Challenge:** Predict students' test scores based on synthetic provided data.

**Notes:** For this challenge I initially tried building out a Random Forest model following methodically and ML pipeline creation I had learned in a previous data science class. The pipeline was easy to build out and the model was successful, but it's RMSE was in the range of 9 which fell short of many other competitors models.

I then pivoted to using XGBoost. Having never used the model before I was interested in learning more about it. The pipeline stayed the same, with the change just being implementing a different model. Below are the parameters I used and what each one entails.

```
model = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42,
    objective='reg:squarederror',
    eval_metric='rmse'
)
```

| Parameter          | Description                            |
| ------------------ | -------------------------------------- |
| `n_estimators`     | Number of trees                        |
| `learning_rate`    | How often weights are updated          |
| `max_depth`        | Tree depth                             |
| `subsample`        | Fraction of observations for each tree |
| `colsample_bytree` | Fraction of features for each tree     |
| `lambda`           | L2 regularization                      |
| `alpha`            | L1 regularization                      |
