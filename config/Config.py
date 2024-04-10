class Config:
    def __init__(self, config = {}):
        for key in config:
            if key not in self.default_config:
                print("WARN: config contains unexpected key: " + key)
                print("this will be ignored")
        self.data = { key: config[key] if key in config else self.default_config[key]  for key in self.default_config } 

        for key in self.data:
            setattr(self, key, self.data[key])

    def __getitem__(self, item):
        return self.data[item]
    
    def __contains__(self, item):
        return item in self.data
    
    # TODO: this is a bit weird - what would be better?
    def __add__(self, other):
        new_default_config = {k: self[k] if k in self else other[k] for k in set(self.data.keys()).union(set(other.data.keys()))}
        class AdditionResultConfig(Config):
            default_config = new_default_config
        return AdditionResultConfig()
