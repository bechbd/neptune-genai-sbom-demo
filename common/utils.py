from pipe import Pipe
def sink():
    def _sink(generator):
        for item in generator:
            pass
    return Pipe(_sink)