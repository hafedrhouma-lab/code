# About

A package tool for Timeseries Forecasting


* `Owner` - Hafed Rhouma
* `contact` - hafed.rhouma@talabat.com
* `github repository` - 

## Project layout

    ├── mkdocs.yml
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── tests #UnitTests
    ├── timeseries_forecast_tool
    │   ├── calendar_forecast #Model using smoothing line and implemented effects
    │   │   ├── effects #module to handle effects descritpions
    │   │   │   ├── effects.py
    │   │   │   ├── __init__.py
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   ├── interfaces #Interfaces for forecast models
    │   │   ├── __init__.py
    │   │   ├── interfaces.py
    │   ├── mix_forecast #Model mixing calendar and reactive forecast
    │   │   └── __init__.py
    │   ├── reactive_forecast #Model using YoY increase rate
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   ├── smoothing #module handling timeseries smoothing
    │   │   ├── __init__.py
    │   │   └── smooher.py
    │   └── utils #helpers functions
    │       ├── exceptions.py
    │       ├── log.py
    │       ├── _helpers.py
    │   ├── __init__.py
    └── tox.ini

## UML Class Diagram

`TODO`


