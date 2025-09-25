# Timeseries Forecast Tool
`Detailled package description can be found here:`

[Package documentation](https://improved-adventure-355m6mk.pages.github.io/)

#### Package Installation:
-Install package: `pip install .`

## Calendar Forecast

#### How to use:

1) Prepare your timeseries to forecast:
```py
import pandas as pd

df = pd.read_csv('test_dataframe.csv', index_col=None)
df = df[["date_sql", "net_valuated_price"]]

df.columns = ["date_sql", "y"]
```

2) Prepare your model to forecast:

```py
from timeseries_forecast_tool.calendar_forecast import CalendarForecast

smoothing_params = {'span': 500, 'beta': 0.0005, 'damped': 1}
calendar_forecast = CalendarForecast(smoothing_params, n_days=100, effects_names=["holidays"])

calendar_forecast.prepare(df)
calendar_forecast.fit(df)
```

Args:

* ```n_days```: number of days to forecast

* ```smoothing_params```: dictionary of smoothing params (keys must be 'span', 'beta', 'damped')

* ```effects_names```: List of effects names. Possibles values are:
    * `date_of_week_of_month`: Effects due to seasonality
  
    * `holidays`: Effects due to holidays days (file must exist)
  
    * `special_events`: Effects due to special events (file must exist)
    

3) Finally, forecast:
```py
forecast_dataframe = calendar_forecast.predict(df)
```

## Reactive Forecast

1) Prepare your model to forecast:

```py
from timeseries_forecast_tool.reactive_forecast import ReactiveForecast

reactive_forecast = ReactiveForecast(n_days=100)
reactive_forecast.prepare(df)
```
Args:

* ```n_days```: number of days to forecast

2) Finally, forecast:
```py
forecast_dataframe = reactive_forecast.predict(df)