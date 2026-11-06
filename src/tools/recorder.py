import threading


class InvocationRecorder:
    """Thread-safe record of actual tool calls. CrewAI deep-copies tools; the recorder is shared on purpose."""

    def __init__(self):
        self.records = []
        self._lock = threading.Lock()

    def add(self, tool, args, result=None):
        with self._lock:
            self.records.append({"tool": tool, "args": args, "result": result})

    def __deepcopy__(self, memo):
        return self
