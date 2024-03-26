from typing import List, Callable, Tuple, Any

class DelayedFunctions:
    _delayed_functions : List[Tuple[Callable, float, Any, Any]]
    def add(self, function:Callable, priority:float=0., *args, **kwargs):
        if not hasattr(self, '_delayed_functions'):
            self._delayed_functions = [(function, priority, args, kwargs)]
        elif (function, priority) not in self._delayed_functions:
            self._delayed_functions.append((function, priority, args, kwargs))
    def launch(self):
        if hasattr(self, '_delayed_functions'):
            self._delayed_functions.sort(key=lambda element: element[1])
            for (function, priority, args, kwargs) in self._delayed_functions:
                function(*args, **kwargs)
            self._delayed_functions.clear()
    def __init__(self):
        self._delayed_functions = []