###Tennis Prediction
With [data generously maintained by Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp), I built some 
machine learning algorithms to predict upcoming winners from upcoming tennis matches. 
In order to reproduce, please pull the tennis_atp repo from Jeff Sackman's tennis_atp repo.  

https://towardsdatascience.com/making-big-bucks-with-a-data-driven-sports-betting-strategy-6c21a6869171

## Setup 
Pull the [repo](https://github.com/JeffSackmann/tennis_atp) in a location of your choosing. Record the location.

Create virtual environment with using pip or conda. 
For example with pip:
python3 -m pip install -r requirements.txt


## How to Run
Set the pythonpath to the project directory:
export PYTHONPATH=/path/to/tennispredictor/

Activate the virtual environment.

# Clean Data
In tennispredictor/preprocessing, run

python clean_data.py --tennisdir "/path/to/data/tennis_atp" --datadir "/path/to/your/data/directory"

# Not finished, in progress!
