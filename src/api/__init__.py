"""FastAPI service for DA prices forecasting.

POST /forecast:
- receives historical hourly data and target date
- builds features
- runs model inference
- returns 24 hourly predictions

Model bundle is loaded once on startup and reused for all requests.
"""
