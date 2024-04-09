# SeqLP

This repository was created to easily train protein language models based on the API huggingface provides. 

## SETUP

First let's set up the directories to be able to store the output of the jobs in the appropriate folders. If you decide to use the exact same bash script for the jobs, as I did please create an email and set it in the script to receive the job notifications.

```bash
mkdir job_output
mkdir job_error
mkdir NLP_train
mkdir NLP_train/serious_train
mkdir NLP_train/test_train
```

## Installation

Start by creating a virtual environment

```
python3 -m venv .venv

source .venv/bin/activate
```

Then clone the repository and install the dependencies
```
git clone git@github.com:nilshof01/SeqLP.git
cd SeqLP
pip install pdm
pdm install 

```

### Ask yourself: Where do I want to train the model?

I was lucky enough to have a big infrastructure and can train the model on various nodes with different capacities. I decided to use Tesla V100 and usually trained on 4 nodes, which is indicated in the bash script at the top. You definetely need to change the parameter -q if you train from outside of DTU because the corresponding GPU may have a different name in your setup. If you train locally you can delete all lines starting with #.
However, to reduce the amount of space my infrastructure provides modules. If you do not have those you must install them separately with the **EXACT** same version I used. Finally, if you have the corresponding modules available, I would recommend you to put them in your .bashrc file to have them available at all times.


## Training

### Test

For training you will use the script __main__.py which takes command line arguments as inputs. You can simply run the script, after you installed all of your dependencies with the following command.

```
python src.seqlp.__main__.py --command_script_dir <PATH_TO_SCRIPT_WITH_WGET_COMMANDS> --run_name test

```

The script will automatically download the data and start the training. Thus, it depends on a certain input structure where in each line of the script is one command starting with wget. You can download thos scripts from the [Observerd Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) (OAS). As an example I filtered for heavy chains with human as species and received an overview of the data [here](https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/). Click on the button here on the website and the script is automatically downloaded.

### Run the test script to check if the setup is working on your server





