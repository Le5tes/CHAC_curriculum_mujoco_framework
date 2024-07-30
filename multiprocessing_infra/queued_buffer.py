from multiprocessing import Queue

class QueuedBuffer:
    def __init__(self, size, num_processes):
        self.replay_buffers = None
        self.queue = Queue(size)
        self.num_processes = num_processes

    def set_buffers(self, replay_buffers):
        self.replay_buffers = replay_buffers

    def consume_from_queue(self):
        if self.replay_buffers is None:
            raise RuntimeError("QUEUED BUFFER CALLED BUT NO BUFFERS SET")
        processes_done = 0
        while not processes_done == self.num_processes:
            got_item = self.queue.get(timeout=120)
            if got_item is not None:
                transition, level = got_item
                self.replay_buffers[level].add(transition)
            else:
                processes_done += 1

    def add(self, transition, level):
        if self.queue.full():
            print("queue full!!")
        else:
            self.queue.put((transition, level), timeout=120)
    
    def mark_process_done(self):
        self.queue.put(None)