from collections import OrderedDict
from multiprocessing import Pipe, Process

from goal_conditioned_baselines import logger

from goal_conditioned_baselines.logger import debug
import os


class PolicyProcess(Process):
    def __init__(self, args=None):
        debug("init super")
        super().__init__(args=args)
        debug("inited super")
        self.message_pipe_front, self.message_pipe_back = Pipe()
        self.data_pipe_front, self.data_pipe_back = Pipe()

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        debug("PP - run called")
        dims, params, robot, env_config, n_train_batches, queued_buffer = self._args
        from environment.GCB_wrapper import GCB_Wrapper
        from environment.mujoco_env import MujocoEnvironment
        from goal_conditioned_baselines.chac import config
        env = GCB_Wrapper(MujocoEnvironment(robot, env_config, logger), env_config)
        self.queued_buffer = queued_buffer
        self.n_train_batches = n_train_batches
        self.policy = config.configure_policy(dims, params, env)
        self.policy.set_train_mode()

        self.active = True
        self.val = 0
        debug("PP - ready to go")
        self.message_pipe_back.send("ready")
        while self.active:
            self.active = self.listen__()
        
    def listen__(self):
        message = self.message_pipe_back.recv()
        if message == "stop":
            return False
        elif message == "train":
            debug("PP - received train")
            ep_number = self.data_pipe_back.recv()
            self.train__(ep_number)
            return True
        elif message == "update_nets":
            net_data = self.data_pipe_back.recv()
            debug("PP - received update nets")
            self.update_nets__(net_data)
            return True
        else:
            debug("Received unexpected message", message)
        
    def train__(self, ep_number):
        success, eval_data, train_duration = self.policy.train(ep_number, {}, self.n_train_batches, True, self.queued_buffer)
        self.queued_buffer.mark_process_done()
        debug("PP - train done, sending back data")
        self.data_pipe_back.send((success, eval_data, train_duration))

    def update_nets__(self, net_data):
        for i, data in enumerate(net_data):
            actor,critic,predictor = data
            self.policy.layers[i].actor.load_state_dict(actor)
            self.policy.layers[i].critic.load_state_dict(critic)
            self.policy.layers[i].state_predictor.load_state_dict(predictor)
        debug("PP - nets updated")
        self.message_pipe_back.send("nets updated")

    
    def train(self, ep_number):
        self.message_pipe_front.send("train")
        self.data_pipe_front.send(ep_number)

    def wait_for_data(self):
        out = self.data_pipe_front.recv()
        debug("PPF - returning data")
        return out
    
    def wait_for_message(self):
        message = self.message_pipe_front.recv()
        debug("PPF - returning message", message)
        return message
    
    def update_nets(self,net_data):

        self.message_pipe_front.send("update_nets")
        self.data_pipe_front.send(net_data)
        
    def stop(self):
        self.message_pipe_front.send("stop")
        self.join()