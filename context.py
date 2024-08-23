# context_manager.py

class ContextManager:
    _instance = None
    def __init__(self) -> None:
        pass
    def __new__(cls, *args, **kwargs):
        if  cls._instance == None:
            cls._instance = object.__new__(cls)
            cls._instance._context = {}
        return cls._instance

    def set(self, data_dict, key=None):
        if not isinstance(data_dict, dict):
            raise ValueError("Input must be a dictionary")
        if key:
            self._context[key] = data_dict
        else:
            self._context.update(data_dict)

    def get(self, key=None, default=None):
        if key is None:
            return self._context
        return self._context.get(key, default)

class Context:
    def __init__(self) -> None:
        self.data = {}
    def update(self,data):
        self.data = data

com = Context()