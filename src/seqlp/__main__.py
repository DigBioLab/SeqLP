from .setup.train_model import TrainModel
from .config import model_config, train_config
import torch
from setup.train_model import TrainModel
from setup.tokenizer import TokenizeData
import argparse
import os


if __name__ == "__main__":
    ####ARGS####
    parser = argparse.ArgumentParser()
    parser.add_argument("--command_script_dir", type=str, required = True, help = "Path to the directory containing the shell scripts to download the data")
    parser.add_argument("--store_dir", type=str, default=None)
    parser.add_argument("--max_file_num", type=int, default=1000000, help = "Maximum number of files to download")
    parser.add_argument("--save_single_csv", type=bool, default=False, help = "Save the single csvs")
    parser.add_argument("--extra_model_config", type=dict, default={}, help = "Extra model configuration. Has to be inputted as a dictionary")
    parser.add_argument("--extra_train_config", type=dict, default={}, help = "Extra training configuration. Has to be inputted as a dictionary")
    parser.add_argument("--use_existing_data", type=bool, default=True, help = "Use existing data and skip downloading. If you enter a path to an existing file you can continue with that.")
    parser.add_argument("--model_type", type=str, default="distilBert", help = "Type of model to use")
    
    args = parser.parse_args()
    store_dir = args.store_dir
    command_script_dir = args.command_script_dir
    max_file_num = args.max_file_num
    save_single_csv = args.save_single_csv
    extra_model_config = args.extra_model_config
    extra_train_config = args.extra_train_config
    ####END ARGS####
    
    tokenize = TokenizeData()
    if args.use_existing_data and store_dir is not None and not os.path.isfile(args.use_existing_data):
        if args.use_existing_data:
            print("You must provide the path to store_dir if you want to use existing data! Continue to download data.")
        filename = tokenize.download_and_prepare(download_commands_script=command_script_dir,
                                                limit = max_file_num,
                                                save_single_csvs = save_single_csv,
                                                user_dir = store_dir,
                                                prep_data_type="uniform")
    else:
        if os.path.isfile(args.use_existing_data):
            filename = args.use_existing_data
        else:
            filename = os.path.join(store_dir,"train_data", "concatenated.csv.gz")
            
    train_encodings, val_encodings = tokenize.tokenize(filename)

    config_model = model_config.get_config(**extra_model_config)
    config_train = train_config.get_config(**extra_train_config)

    if torch.cuda.isavailable():
        config_train["no_cuda"] = False
        
    Train = TrainModel(train_encodings,
                        val_encodings,
                        model_config=config_model,
                        train_params=config_train,
                        user_dir=store_dir)
    Train.train()