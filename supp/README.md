## Feature selection benchmarking experiments

All the code and the results of the experiment can be found in fsTest.

#### Reproducing the results

The results can be fully reproduced once `celery` and `RabbitMQ`  is installed and set up. Follow [this](https://tests4geeks.com/python-celery-rabbitmq-tutorial/) tutorial to install these requirements. 

Install the required Python packages:
- `scikit-learn`
- `numpy`
- `scipy`
- `boruta`
- `pandas`
- `seaborn`
- `matplotlib`
- `bottleneck`

Then simply go to the /supp folder and in one terminal and execute:
`celery -A fsTest worker --loglevel=info`

Then from the same folder, open another terminal and do:
`python2 -m fsTest.run_tasks`

#### Reproducing the benchmarking figures and tables

The figures and tables could be recreated using the `merge_results.py` and `merge_results_topvar.py` scripts. The input and output folder variables has to be changed as explained by the comments. 

----------------

## Running CorrMapper's pipeline on simulated datasets and comparing it with mixOmics and marginal correlation networks

Similarly to the feature selection experiment go into the `supp` folder, then execute in one terminal:
`celery -A cmTest worker --loglevel=info`

Then from the same folder, open another terminal and do:
`python2 -m cmTest.run_tasks`

#### Reproducing the benchmarking figures and tables
Run `merge_results.py` and `merge_results2.py`.
