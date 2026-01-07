import time

class Stopwatch:
    """
    A simple stopwatch class to measure elapsed time.
    """
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        """
        Start or resume the stopwatch.
        """
        if not self.running:
            self.start_time = time.time() - self.elapsed_time
            self.running = True

    def stop(self):
        """
        Stop the stopwatch and record the elapsed time.
        """
        if self.running:
            self.elapsed_time = time.time() - self.start_time
            self.running = False

    def reset(self):
        """
        Reset the stopwatch to zero.
        """
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def get_elapsed_time(self):
        """
        Get the elapsed time in seconds.
        
        Returns:
            float: The elapsed time in seconds.
        """
        if self.running:
            return time.time() - self.start_time
        return self.elapsed_time