import inspect
import pdb

class Serializable:

    def __init__(self):
        pass
        
    def record_args(self, locals_):
        spec = inspect.getfullargspec(self.__init__)

        ## **kwargs of init method
        # print('\n', locals_)
        # print(spec.varkw)
        # print(spec)
        if spec.varkw:
            kwargs = locals_[spec.varkw].copy()
        else:
            kwargs = dict()

        ## kwargs not set at instantiation;
        ## set to defaults
        for key in spec.kwonlyargs:
            kwargs[key] = locals_[key]

        ## *args of init method
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()

        ## init args after self
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.args_ = tuple(in_order_args) + varargs
        self.kwargs_ = kwargs
        