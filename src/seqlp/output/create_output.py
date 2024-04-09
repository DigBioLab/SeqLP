from tabulate import tabulate
import csv
import os
import time

class GenerateOutput:
    def __init__(self, save_dir, run_name) -> None:
        
        self.save_dir = save_dir
        self.run_name = run_name
        self.start_time = self.start_timer()
    
    @staticmethod
    def save_table_as_csv(save_path:str, given_dict:dict):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])
            for key, value in given_dict.items():
                writer.writerow([key, value])
    
    def table_config(self, config:dict, type_:str):
        table = tabulate(list(config.items()), headers = ["Parameter", "Value"])
        save_path = os.path.join(self.save_dir, f'config_{type_}_{self.run_name}.csv')
        self.save_table_as_csv(save_path, config)
        return table

    @staticmethod
    def start_timer():
        return time.time()
    
    @staticmethod
    def stop_timer():
        return time.time()
    
    def time_dif(self,  ):
        stop = self.stop_timer()
        elapsed_time = stop - self.start_time
        elapsed_time_minutes = elapsed_time / 60
        print(f"Elapsed time: {elapsed_time_minutes} minutes")
        return int(elapsed_time_minutes)
    
    def collect_general_info(self, num_sequences:int, elapsed_time_minutes:int, num_parameters:int,  model_type:str):
        general_info = {}
        general_info["Number of sequences"] = num_sequences
        general_info["Run Time"] = elapsed_time_minutes
        general_info["Num parameters"] = num_parameters
        general_info["Name_run"] = self.run_name
        general_info["Model"] = model_type
        table = tabulate(list(general_info.items()), headers = ["Parameter", "Value"])
        save_path = os.path.join(self.save_dir, f'general_run_info_{self.run_name}.csv')
        self.save_table_as_csv(save_path, general_info)
        return table
        
    
    