from setup.train_model import TrainModel
from config.model_config import SetupModelConfig
from config.train_config import SetupTrainConfig
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
    parser.add_argument("--max_sequence_num", type=int, default=1000000, help = "Maximum number of Sequences to include in training")
    parser.add_argument("--save_single_csv", type=bool, default=False, help = "Save the single csvs")
    parser.add_argument("--extra_model_config", type=str, default="", help = "Extra model configuration. Has to be inputted as a dictionary")
    parser.add_argument("--extra_train_config", type=str, default="", help = "Extra training configuration. Has to be inputted as a dictionary")
    parser.add_argument("--use_existing_data", type=str, default="", help = "Use existing data and skip downloading. If you enter a path to an existing file you can continue with that.")
    parser.add_argument("--model_type", type=str, default="distilBert", help = "Type of model to use")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help = "Probability of masking a token")
    parser.add_argument("--max_length", type=int, default=150, help = "Maximum length of the sequences")
    
    args = parser.parse_args()
    store_dir = args.store_dir
    command_script_dir = args.command_script_dir
    max_file_num = args.max_sequence_num
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
            
    train_encodings, val_encodings = tokenize.tokenize(filename, max_length=args.max_length, mlm_probability=args.mlm_probability)
    
    Config_model_setup = SetupModelConfig()
    if os.path.isfile(extra_model_config):
        model_config_dict = Config_model_setup.read_json(extra_model_config)
        config_model = Config_model_setup.get_config(tokenize,**model_config_dict)
    else:
        Config_model_setup.set_tokenize_vocab(tokenize)
        Config_model_setup.set_max_len(args.max_length)
        config_model = Config_model_setup.config
        print("Using default model configuration")
    if os.path.isfile(extra_train_config):
        train_config_dic = SetupTrainConfig.read_json(extra_train_config)
        config_train = SetupTrainConfig().get_config(**train_config_dic)
    else:
        config_train = SetupTrainConfig().config
        print("Using default training configuration")


    if torch.cuda.is_available():
        config_train["use_cpu"] = False
    else:
        config_train["use_cpu"] = True
        
    Train = TrainModel(train_encodings,
                        val_encodings,
                        tokenize.data_collator,
                        model_config=config_model,
                        train_params=config_train,
                        model_type=args.model_type,
                        user_dir=store_dir)
    Train.train()