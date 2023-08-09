# The hidden impacts of rising interest rates

This repository contains the data and scripts powering
[The hidden impacts of rising interest rates]() page.

Methodology can be found here: [Methodology]().

To reproduce the analysis, `Python >= 3.10` is required.
Package requirements can be found in `requirements.txt`.

### Repository structure


The repository contains the following subfolders:
- `output`: contains the output of the analysis including 
    key numbers and tables used to generate the charts.
- `raw_data`: contains the raw data used in the analysis.
- `scripts`: contains the scripts used to generate the analysis. 

The `visualization` subfolder contains the scripts to generate the charts.
run `update_visualizations.py` to generate to update the chart data. 
This script also contains functions to generate individual chart data.
`interest_observable.py` contains the functions to create the 
interactive interest payment chart hosted on Observable.
`interest_flourish.py` contains the functions to create the
charts hosted on Flourish.

The `debt` subfolder contains the scripts to 
get debt data and calculate interest payments. The
`fed_rates` subfolder contains the scripts to get
the federal rate hike chart. The `government` subfolder
contains the scripts to get the government spending and GDP data.
The `inflation` subfolder contains the scripts to get the inflation data,
calculate aggregate inflation and produce infation key numbers.
The `social spending` subfolder contains the scripts
get the education and health expenditure data and 
produce debt and health chart.
