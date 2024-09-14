import numpy as np
import params as p


class BaseEnv():
    done:bool     = True
    max_steps:int = None
    def __init__(self) -> None:
        ...
    def step(self, action:int)->tuple[np.ndarray, float, bool, dict]:
        ...
    def reset(self)->np.ndarray:
        ...
    def render(self)->np.ndarray:
        ...
    def close(self)->None:
        ...
    def set_params(self, param:p.EnvParams):
        """将param中的参数赋予本环境"""
        if isinstance(param, p.BaseParams):
            for key in dir(param):
                if not callable(getattr(param, key)) and not key.startswith('__'):
                    setattr(self, key, getattr(param, key))