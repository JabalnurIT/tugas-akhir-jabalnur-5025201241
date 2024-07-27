delete_dir(f'{config["USED_DIR"]}{self.current_model}')
self.current_model = "none"
self.model_list = get_dir_list(config["USED_DIR"])