"""FastAPI inference service.

Контракт inference:
    POST /forecast — клиент шлёт history (hourly master frame) + as_of +
    target_date → сервер делает FE через src.features, грузит current bundle,
    возвращает 24 hourly preds на target_date.

Bundle загружается один раз при startup (lifespan), не на каждый запрос.
"""
