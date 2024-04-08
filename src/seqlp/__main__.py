from setup.train_model import TrainModel
from config.model_config import SetupModelConfig
from config.train_config import SetupTrainConfig
import torch
from setup.train_model import TrainModel
from setup.tokenizer import TokenizeData
import argparse
import os
from output.create_output import GenerateOutput
from output.memory_usage import print_gpu_utilization

if __name__ == "__main__":
    ####ARGS####
    parser = argparse.ArgumentParser()
    parser.add_argument("--command_script_dir", type=str, required = True, help = "Path to the directory containing the shell scripts to download the data")
    parser.add_argument("--run_name", type =str, required = True)
    parser.add_argument("--store_dir", type=str, default=None)
    parser.add_argument("--max_sequence_num", type=int, default=1000000, help = "Maximum number of Sequences to include in training")
    parser.add_argument("--save_single_csv", type=bool, default=False, help = "Save the single csvs")
    parser.add_argument("--extra_model_config", type=str, default="", help = "Extra model configuration. Has to be inputted as a dictionary")
    parser.add_argument("--extra_train_config", type=str, default="", help = "Extra training configuration. Has to be inputted as a dictionary")
    parser.add_argument("--use_existing_data",nargs = "+", type=str, default=[], help = "Use existing data and skip downloading. If you enter a path to an existing file you can continue with that.")
    parser.add_argument("--model_type", type=str, default="distilBert", help = "Type of model to use")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help = "Probability of masking a token")
    parser.add_argument("--max_length", type=int, default=150, help = "Maximum length of the sequences")
    
    args = parser.parse_args()
    store_dir = args.store_dir
    command_script_dir = args.command_script_dir
    max_seq_num = args.max_sequence_num
    save_single_csv = args.save_single_csv
    extra_model_config = args.extra_model_config
    extra_train_config = args.extra_train_config
    run_name = args.run_name
    
    ####END ARGS####
    if store_dir is None:
        
        os.mkdir(run_name)
        store_dir = os.path.abspath(run_name)
    else:
        path = os.path.join(store_dir, run_name)
        if os.path.isdir(path) == False:
            os.mkdir(path)
        store_dir = os.path.join(store_dir, run_name)
        
    Output = GenerateOutput(store_dir, run_name)
    print_gpu_utilization()
    tokenize = TokenizeData()
    
    if len(args.use_existing_data) ==0 and args.store_dir is not None:
        print(f"Initialize Download from {command_script_dir}")
        train_filename, val_filename = tokenize.download_and_prepare(download_commands_script=command_script_dir,
                                                                     run_name=run_name,
                                            limit = max_seq_num,
                                            save_single_csvs = False,
                                            user_dir = store_dir)
    else:
        if os.path.isfile(args.use_existing_data[0]) and os.path.isfile(args.use_existing_data[1]):
            train_filename = args.use_existing_data[0]
            val_filename = args.use_existing_data[1]
        else:
            FileExistsError("The files you provided do not exist. Please check the paths.")
    train_encodings= tokenize.tokenize(train_filename)
    print_gpu_utilization()
    val_encodings = tokenize.tokenize(val_filename)
    print_gpu_utilization()

    print("Data Collection and Tokenizing successful")
    
    torch.cuda.empty_cache()
    Config_model_setup = SetupModelConfig()
    if os.path.isfile(extra_model_config):
        model_config_dict = Config_model_setup.read_json(extra_model_config)
        config_model = Config_model_setup.get_config(tokenize,**model_config_dict)
        print(config_model)
    else:
        Config_model_setup.set_tokenize_vocab(tokenize)
        Config_model_setup.set_max_len(args.max_length)
        config_model = Config_model_setup.config
        print("Using default model configuration")
    if os.path.isfile(extra_train_config):
        train_config_dic = SetupTrainConfig.read_json(extra_train_config)
        config_train = SetupTrainConfig().get_config(**train_config_dic)
        print(config_train)
    else:
        config_train = SetupTrainConfig().config
        print("Using default training configuration")
    print_gpu_utilization()

    if torch.cuda.is_available():
        print("CUDA is available. GPU support is enabled.")
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
    print_gpu_utilization()

    print("Model setup successful")
    Train.train()
    num_parameters = Train.model.num_parameters()
    print(Output.table_config(config_model, "model"))
    print(Output.table_config(config_train, "train"))
    time_dif = Output.time_dif()
    print(Output.collect_general_info(max_seq_num,
                                      time_dif,
                                      num_parameters,
                                      args.command,
                                      args.model_type))
    Train.trainer.save_model(os.path.join(store_dir, "model"))
    print("Execution successful")