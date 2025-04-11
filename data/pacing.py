import yaml

class PacingFunction:
    def __init__(self, config_path='configs/pacing_functions.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_function(self, name='linear'):
        if name == 'linear':
            return self.linear_pacing
        elif name == 'exponential':
            return self.exponential_pacing
        elif name == 'step':
            return self.step_pacing
        else:
            raise ValueError(f"Unknown pacing function: {name}")
    
    def linear_pacing(self, iteration):
        cfg = self.config['pacing_functions']['linear']
        progress = min(iteration / cfg['steps'], 1.0)
        return cfg['start'] + (cfg['end'] - cfg['start']) * progress
    
    def exponential_pacing(self, iteration):
        cfg = self.config['pacing_functions']['exponential']
        return min(cfg['start'] * (cfg['gamma'] ** (-iteration)), cfg['end'])
    
    def step_pacing(self, iteration):
        cfg = self.config['pacing_functions']['step']
        value = cfg['values'][0]
        for milestone, val in zip(cfg['milestones'], cfg['values'][1:]):
            if iteration >= milestone:
                value = val
        return value