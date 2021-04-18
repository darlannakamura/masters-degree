class method:
    def __init__(self, m):
        self._m = m

    def __call__(self, *args, **kwargs):
        return self._m(*args, **kwargs)

    @classmethod
    def methods(cls, subject):
        def g():
            for name in dir(subject):
                _method = getattr(subject, name)
                if isinstance(_method, method):
                    yield name, _method
        return {name: _method for name, _method in g()} 
