# SeqLP

This repository was created to easily train protein language models based on the API huggingface provides. 

## SETUP

First let's set up the directories to be able to store the output of the jobs in the appropriate folders. If you decide to use the exact same bash script for the jobs, as I did please create an email and set it in the script to receive the job notifications.


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

### Further dependencies

cuda 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

ffmpeg 4.2.2: https://ffmpeg.org/

cudnn 11.8: https://developer.nvidia.com/rdp/cudnn-archive

gcc 8.4.0: https://gcc.gnu.org/gcc-8/


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
Alternatively, you can train the model on nanobodies. The data can be downloaded from the [Nanobody Database](https://research.naturalantibody.com/nanobodies). If you want to use those you should prepare the data appropriately. You can download the file which is provided there under NGS sequences and use the methods which are implemented in this repository.

```python
from src.seqlp.setup.data_prep import FromTSV, Prepare

sequence_series = FromTSV.read_tsv(filepath, limit = <LIMIT_NO_SEQUENCES>)
sequence_series = Prepare.drop_duplicates(sequence_series)
sequence_series = Prepare.insert_space(sequence_series) # very important otherwise the tokenizing will not work appropriately.
train_sequences, val_sequences = Prepare.create_train_test(sequence_series)
train_sequences.to_csv("train.csv", index = False)
val_sequences.to_csv("val.csv", index = False)

```


### Load the trained model

```
from src.seqlp.visualize.load_model import DataPipeline

sequences = ["<SEQUENCE_1>", "<SEQUENCE_2>"] # No spaces between letters,here. Function does this automatically.
Setup = LoadModel(<PATH_TO_MODEL>)
embeddings = Setup._get_encodings(sequences)
embeddings_array = embeddings.to_numpy()
```




