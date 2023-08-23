# Tennis Prediction
With [data generously maintained by Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp), I built some 
machine learning algorithms to predict upcoming winners from upcoming tennis matches. 
In order to reproduce, please pull the tennis_atp repo from Jeff Sackman's tennis_atp repo.  

## Setup 
Pull the [repo](https://github.com/JeffSackmann/tennis_atp) in a location of your choosing. Record the location.


## How to Run
Set the pythonpath to the project directory:
    export PYTHONPATH=/path/to/tennispredictor/

Activate the virtual environment.
Create virtual environment with using pip or conda. 
For example with pip:
    python -m pip install -r requirements.txt


## Clean Data
The clean data script consolidates CSVs, removes null values, converts strings to numeric and calculates an ELO score, which
from the literature is more predictive than tennis ranking. 

    python preprocessing/clean_data.py --tennisdir "/path/to/data/tennis_atp" --datadir "/path/to/your/data/directory"

## Train Model
To train the MLP

    python main/train.py --csv "/path/to/your/data/directory/atp_database.csv"

## Classifiers
Traditional classifiers include Nearest Neighbors, Linear SVM, Gaussian Process, Decision Tree,
Random Forest, Neural Net, AdaBoost, Naive Bayes, and QDA.

To train traditional classifiers on the data, run

    python main.classifier.py --csv "/path/to/your/data/directory/atp_database.csv"  
    --rootdir "/path/to/your/data/directory"

where `rootdir` is your results directory. 

The classifiers are saved based on `timestring`.  

To predict on your trained classifiers, run

    python main.classifier.py --csv "/path/to/your/data/directory/atp_database.csv"  
        --rootdir "/path/to/your/data/directory" 
        --timestring "timestring of your trained classifier"
        --classifier_name "Name of your classifier"

## Deploy on New Data
To predict on new matches, run 

    python preprocessing/generate_deploy.py

This script saves a file called `deploy.csv` which can be used in lieu of 
atp_database.csv.

To deploy on a classifier, replace csv with the output of generate_deploy.py.  

To deploy on the neural network, run

    python main/deploy.py --csv /path/to/csv
    --ckpt_path /path/to/model
    --rootdir /path/to/your/analysis/dir

## Results

### Classifiers

Test Set Scores:
- Linear SVM 0.9275
- Decision Tree 0.9207
- Random Forest 0.8714
- AdaBoost 0.926
- Neural Net 0.9003
- Nearest Neighbors 0.7149
- Naive Bayes 0.9304
- QDA 0.9166

The classifiers scored between 90-92% accuracy on the test set save for the Nearest Neighbors
classifier, perhaps due to the high dimensionality of the feature space. 

Analyzing feature importances with permutations and Mean Accuracy Decrease, scoring with
Negative Mean Squared Error, it seems winning streaks and losing streaks were critical across
all classifiers. ELO, handedness, and seed affected some classifiers. 

![Naive Bayes Feature Importance Figure](/results/AdaBoost_neg_mean_squared_error_importance.png)