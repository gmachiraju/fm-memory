import pickle

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)    