## Literature Review / Sources Overview

This section summarizes the key data sources and academic background used in the thesis, as well as the methodological foundations of electricity price forecasting.

### Data Sources

**ENTSO-E Transparency Platform**  
- Hourly day-ahead electricity prices for the Netherlands  
- System load data  
- Primary source for the target variable and market-related features  

**KNMI (Royal Netherlands Meteorological Institute)**  
- Official meteorological observations for the Netherlands  
- Daily average temperature (TG) from the De Bilt weather station  
- Used as an exogenous factor influencing electricity demand  

**TTF (Title Transfer Facility)**  
- Natural gas price data from the main European gas trading hub  
- Used as an indicator of marginal generation costs in the Dutch electricity market  

### Methodological Background

The literature review covers the following groups of methods commonly applied to electricity price forecasting:

- **Classical time series models**  
  ARIMA and SARIMA models used as statistical baselines for short-term forecasting  

- **Time series decomposition methods**  
  STL and MSTL approaches for handling multiple seasonalities in hourly electricity price data  

- **Machine learning and hybrid approaches**  
  Models combining statistical structure with machine learning to capture nonlinear effects and exogenous influences  

### Scope of Review

The reviewed sources address:
- electricity market structure and price formation mechanisms;
- properties of hourly electricity price data (volatility, seasonality, price spikes);
- strengths and limitations of different forecasting approaches;
- practical considerations for short-term (day-ahead) forecasting.
