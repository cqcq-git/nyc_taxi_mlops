# Drift Detection Runbook

## When to Run

- Weekly as a scheduled monitoring job
- After data pipeline changes
- After model retraining
- When prediction metrics shift unexpectedly

## How to Interpret Results

**No drift detected:** Continue normal operation.

**Moderate drift:** Investigate which features changed. Check upstream data quality and recent business/seasonal changes.

**Severe drift:** Consider retraining and review whether the current model is still appropriate.

## Common Causes

- Seasonal taxi demand changes
- Different pickup/dropoff zone patterns
- Weather or events
- Data collection changes
- Feature engineering changes

## Actions

**Data drift only:** Investigate input distribution changes. Retraining may be needed.

**Prediction drift only:** Investigate model behavior and prediction distribution.

**Both data and prediction drift:** Strong signal to retrain and validate a new model.
