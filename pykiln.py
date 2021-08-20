import sys
import inspect
import os
from abc import ABC, abstractmethod

_processManager = None

HASH_MAX = 2 ** 64
def _hash_join(old, val):
    ret = (hash(val) * 417023) % HASH_MAX
    ret = (ret + old * 13691) % HASH_MAX
    ret = (ret + 1692563) % HASH_MAX
    return ret

def _str_hash(h):
    h = _get_hash(h)
    return ('0' * 16 + hex(h)[2:])[-16:]

def _get_hash(obj):
    ret = _get_hash_inner(obj)
    return ret

def _get_hash_inner(obj):
    ret = 0
    if isinstance(obj, list) or isinstance(obj, tuple):
        ret = _hash_join(ret, 11)
        for item in obj:
            ret = _hash_join(ret, _get_hash_inner(item))
    elif isinstance(obj, dict):
        ret = _hash_join(ret, 13)
        for key, item in obj.items():
            ret = _hash_join(ret, _get_hash_inner(key))
            ret = _hash_join(ret, _get_hash_inner(item))
    elif isinstance(obj, float):
        obj = obj // 1e-10
        ret = int(obj)
    elif isinstance(obj, str):
        # hash() of str changed every runtime
        ret = _hash_join(ret, 19)
        ret = _hash_join(ret, len(obj))
        for c in obj:
            ret = _hash_join(ret, ord(c))
    else:
        try:
            ret = _hash_join(ret, hash(obj))
        except TypeError:
            raise TypeError("Cannot make hash")
    return ret
        
def _get_args(sig, fn_args, fn_kwargs):
    dep_arg = {}
    for key, param in list(sig.parameters.items()):
        if param.kind == param.POSITIONAL_ONLY:
            # if key in fn_kwargs:
            #     raise TypeError("%s() got some positional-only arguments passed as keyword arguments: '%s'" % (fn_name, key))
            if len(fn_args) == 0:
                raise TypeError("%s() missing 1 required positional argument: '%s'" % (fn_name, key))
            dep_arg[key] = fn_args[0]
            fn_args = fn_args[1:]
        elif param.kind == param.POSITIONAL_OR_KEYWORD:
            if key in fn_kwargs:
                dep_arg[key] = fn_kwargs[key]
                del fn_kwargs[key]
            elif len(fn_args) > 0:
                dep_arg[key] = fn_args[0]
                fn_args = fn_args[1:]
            elif not(param.default is param.empty):
                dep_arg[key] = param.default
            else:
                raise TypeError("%s() missing 1 required positional argument: '%s'" % (fn_name, key))
        elif param.kind == param.VAR_POSITIONAL:
            dep_arg[key] = tuple(fn_args)
            fn_args = []
        elif param.kind == param.KEYWORD_ONLY:
            if key in fn_kwargs:
                dep_arg[key] = fn_kwargs[key]
                del fn_kwargs[key]
            else:
                raise TypeError("%s() missing 1 required keyword-only argument: '%s'" % (fn_name, key))
        elif param.kind == param.VAR_KEYWORD:
            dep_arg[key] = fn_kwargs
            fn_kwargs = {}
    if len(fn_args) > 0:
        raise TypeError("Too many arguments")
    if len(fn_kwargs) > 0:
        raise TypeError("Cannot pass paramater %s to this function" % fn_kwargs.keys()[0])
    return dep_arg

class Serializer(ABC):
    @abstractmethod
    def serialize(self, fname, obj):
        raise NotImplementedError()
    
    @abstractmethod
    def deserialize(self, fname):
        raise NotImplementedError()

class PickleSerializer(Serializer):
    def serialize(self, fname, obj):
        import pickle
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
    
    def deserialize(self, fname):
        import pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)
        
class StubSerializer(Serializer):
    def serialize(self, fname, obj):
        pass
    def deserialize(self, fname):
        return None
    
serializers = {
    'pickle': PickleSerializer(),
    'stub': StubSerializer()
}

def register(tmpdir = None, exclude_params = [], filetype = 'pickle', 
        hash_func = None, expire_func = None):
    """
    Register a function to Pykiln.
    '''
    @pykiln.register()
    def my_func():
        pass
    '''
    """
    def wrapper(fn):
        nonlocal tmpdir
        if not inspect.isfunction(fn):
            raise TypeError("Given object is not a function.")
        if not (filetype in serializers):
            raise TypeError("Unsupported filetype %s" % filetype)
        
        # Get information about fn
        fn_name = fn.__name__
        
        # Configure tmpdir
        if tmpdir == None:
            try:
                fn_fname = inspect.getsourcefile(fn)
            except TypeError:
                raise TypeError("Cannot register builtin function")
            tmpdir = os.path.join(os.path.dirname(fn_fname),
                    "__pykiln_tempdir__",
                    os.path.basename(fn_fname),
                    fn_name)
            
        os.makedirs(tmpdir, exist_ok=True)
        
        # Check arguments of fn
        sig = inspect.signature(fn)
        for k in exclude_params:
            if not(k in list(sig.parameters.keys())):
                raise NameError("Paramater %s in exclude_params not found" % k)
        
        def inner_fn(*fn_args, **fn_kwargs):
            # List arguments
            fname, ret = _update_sync(inner_fn, fn_args, fn_kwargs)
            if fname == None:
                return ret
            else:
                ss = serializers[os.path.splitext(fname)[1][1:]]
                return ss.deserialize(fname)
                
        # name the original function to make it pickle-able
        orig_fn_name = fn_name + "__pykiln_origfn"
        fn.__qualname__ = fn.__qualname__.replace(fn_name, orig_fn_name)
        fn.__name__ = orig_fn_name
        setattr(sys.modules[fn.__module__], orig_fn_name, fn)
        
        inner_fn._tmpdir = tmpdir
        inner_fn._fn = fn
        inner_fn._exclude_params = exclude_params
        inner_fn._sig = sig
        inner_fn._filetype = filetype
        inner_fn._hash_func = hash_func
        inner_fn._expire_func = expire_func
        return inner_fn
    return wrapper

def _get_filename(fn, fn_args, fn_kwargs):
    dep_arg = _get_args(fn._sig, fn_args, fn_kwargs)
    for p in fn._exclude_params:
        if p in dep_arg:
            del dep_arg[p]
    
    if fn._hash_func == None:
        h = _str_hash(dep_arg)
    else:
        h = 13
        for key, val in dep_arg.items():
            h = _hash_join(h, _get_hash(key))
            h = _hash_join(h, fn._hash_func(val, _get_hash))
        h = _str_hash(h)
    checked = False
    for ft in serializers.keys():
        fname = os.path.join(fn._tmpdir, h + "." + ft)
        if not os.path.isfile(fname): continue
        if fn._expire_func == None or not fn._expire_func(fname):
            checked = True
            break
    else:
        fname = os.path.join(fn._tmpdir, h + "." + fn._filetype)
    return (fname, checked)

def _update_sync(fn, fn_args, fn_kwargs):
    try:
        inner_fn = fn._fn
    except AttributeError:
        # normal function
        ret = fn(*fn_args, **fn_kwargs)
        return (None, ret)
    fname, checked = _get_filename(fn, fn_args, fn_kwargs)
    if _processManager != None:
        if _processManager.exists(fname):
            return _processManager.wait_for(fname)
    if checked or os.path.isfile(fname) and (fn._expire_func == None or not fn._expire_func(fname)):
        _update_atime(fname)
        return (fname, None)
    # Update required
    ret = inner_fn(*fn_args, **fn_kwargs)
    ss = serializers[os.path.splitext(fname)[1][1:]]
    ss.serialize(fname, ret)
    return (None, ret)

def _update_async(fn, fn_args, fn_kwargs):
    global _processManager
    try:
        inner_fn = fn._fn
        is_special = True
    except AttributeError:
        inner_fn = fn
        is_special = False
    if is_special:
        fname, checked = _get_filename(fn, fn_args, fn_kwargs)
        if _processManager != None:
            if _processManager.exists(fname):
                return
        if checked or os.path.isfile(fname) and (fn._expire_func == None or not fn._expire_func(fname)):
            _update_atime(fname)
            return
    else:
        i = 0
        if _processManager != None:
            i = _processManager.get_len()
        fname = "%d.stub" % i
    # Update required
    if _processManager == None:
        _processManager = ProcessManager()
    _processManager.join(fname, inner_fn, fn_args, fn_kwargs)
    
def _update_atime(fname):
    from datetime import datetime
    atime = datetime.now().timestamp()
    os.utime(fname, (atime, atime))
    
def background(func, *args, **kwargs):
    r"""
    >>> @pykiln.register
    >>> def calc(a, b, c):
    >>>     return a * b + c
    >>> 
    >>> def func():
    >>>     a = b = 1
    >>> 
    >>>     # Generate in background
    >>>     results = [pykiln.background(calc, a, b, c) for c in range(10)]
    >>> 
    >>>     # Using the calculation results
    >>>     for result in results:
    >>>         print(result.get()) # print calculation of a * b + c
    """
    _update_async(func, args, kwargs)
    class BackgroundTask:
        def __init__(self, fn, fn_args, fn_kwargs):
            self._fn = fn
            self._fn_args = fn_args
            self._fn_kwargs = fn_kwargs
            self.cached = False
            self.result = None
            
        def get(self):
            fname, ret = _update_sync(self._fn, self._fn_args, self._fn_kwargs)
            if fname == None:
                return ret
            elif self.cached:
                return self.result
            else:
                ss = serializers[os.path.splitext(fname)[1][1:]]
                self.cached = True
                self.result = ss.deserialize(fname)
                return self.result
    return BackgroundTask(func, args, kwargs)

def wait_background():
    if _processManager != None:
        _processManager.wait_all()

import concurrent
from concurrent.futures import ProcessPoolExecutor
class ProcessManager:
    def __init__(self, *args, **kwargs):
        self.processes = {}
        self.exe = ProcessPoolExecutor(*args, **kwargs)
    
    def get_len(self):
        return len(self.processes)
    
    def exists(self, fname):
        if fname in self.processes:
            if not self.processes[fname].done():
                return True
        return False
    
    def wait_for(self, fname):
        global _processManager
        if fname in self.processes:
            ret = self.processes[fname].result()
            ext = os.path.splitext(fname)[1][1:]
            ss = serializers[ext]
            ss.serialize(fname, ret)
            del self.processes[fname]
            if len(self.processes) == 0:
                _processManager = None
            return (None, ret)
        return (fname, None)
    
    def wait_all(self):
        while len(self.processes) > 0:
            fname = list(self.processes.keys())[0]
            self.wait_for(fname)
    
    def join(self, fname, fn, fn_args, fn_kwargs):
        self.processes[fname] = self.exe.submit(fn, *fn_args, **fn_kwargs)
        
    def __enter__(self):
        return self
    
    def __exit__(self):
        self.__del__()
        return False
    
    def __del__(self):
        self.exe.shutdown(wait=False)
