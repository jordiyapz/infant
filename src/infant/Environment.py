import socketio
import requests
import torch
import urllib
from torchvision.io import decode_png
from time import sleep
import asyncio
import numpy as np
import multiprocessing as mp

from infant.util import prefer_gpu

def data_uri_to_state(data_uri, device=None):
    '''Convert data_uri to state.'''
    device = device or prefer_gpu()
    res = urllib.request.urlopen(data_uri)
    raw_image:bytes = res.file.read()
    tensor = torch.frombuffer(np.array(raw_image), 
                              dtype=torch.uint8)
    tensor.to(device=device)
    state = decode_png(tensor)[:3]  # iqnore last alpha channel as it always full

    return state

def post_process_action(action):
    return dict(zip(('x', 'y', 'down'), map(float, action)))

class SimNs(socketio.ClientNamespace):
    def on_disconnect(self):
        print('Disconnected')

class Environment:
    def __init__(self, sim_id:str=None, base_url='http://localhost:3000'):
        self.base_url = base_url
        self.sim_id = sim_id

        self._is_done = False
        self._created_sims = []
        self._on_stop = None
        self._use_step = False
        self._recv_render = {'is_new': False, 'data': None}
        self._init_sent = False
        self._running_loop = None

        self._sock = socketio.Client()
        self.wait = self._sock.wait

        if sim_id:
            self._register_namespace()

        self._episode_ct = 0 
        self._max_episodes = None

        self.prev_state = None
        self.total_reward = 0

        self._train_lock = mp.Lock()
    
    def _register_namespace(self):
        self._ns = SimNs(f'/sim-{self.sim_id}')
        self._sock.register_namespace(self._ns)

    def set_running_loop(self, loop:asyncio.AbstractEventLoop):
        self._running_loop = loop

    def create(self):
        '''Create new simulation'''
        response = requests.post(f'{self.base_url}/sims').json()
        self.sim_id = response['id']
        self._created_sims.append(self.sim_id)
        self._register_namespace()
    
    def connect(self, disconnect_on_done=True, use_step=True):
        '''Connect to socket.io server'''
        self._sock.on('connect', lambda: print(f'Connected to {self._ns.namespace}'))
        self._sock.connect(self.base_url)

        if disconnect_on_done:
            self._on_stop = lambda: self.disconnect()
        
        self._use_step = use_step
        # if use_step:
        #     def on_render(*args):
        #         self._recv_render['is_new'] = True
        #         self._recv_render['data'] = args
        #     self._sock.on('sim:render', on_render, self._ns.namespace)

    def init(self, max_episodes=None):
        '''Send initialization signal to simulation server. 
        In other words, init simulation-loop.'''
        self._is_done = False
        self._max_episodes = max_episodes
        self._sock.emit('sim:init', namespace=self._ns.namespace)
        self._init_sent = True

    def disconnect(self):
        '''Close socket.io connection'''
        self._sock.disconnect()

    def reset_episode_count(self):
        self._episode_ct = 0
    
    def reset(self):
        self.reset_episode_count()
        self.prev_state = None
        self.total_reward = 0

    def step(self, action:torch.Tensor):
        '''[WIP] Step through the simulation. 
        Returns Awaitables that resolves into state.
        '''
        if not self._running_loop:
            raise Exception('Attach event loop first')

        future = self._running_loop.create_future()
        # try:
            # while not self._recv_render['is_new']:
            #     sleep(.01)
        self._sock.on('sim:render', lambda *args: future.set_result(args), self._ns.namespace)
        if not self._init_sent:
            self.init()

        action_dict = dict(zip(('x', 'y', 'down'), map(float, action)))
        self._sock.emit('sim:action', action_dict, self._ns.namespace)
            # resolve(self._recv_render['data'])
        # except Exception as error:
        #     reject(error)

        return future

        # return Promise(get_next)

    def on_state(self, on_state, callbacks=[], device=None):
        '''Attach simulation-loop
        
        Parameters:
            on_state: Function that takes `state`, `reward`, `env` and return `action`.
        
        '''
        
        def on_render(data_uri, reward):
            '''Thin wrapper to decode data_uri and send encoded action.'''
            try:
                state = data_uri_to_state(data_uri, device)

                # Calls user-defined on_state function
                self._train_lock.acquire()
                action:torch.Tensor = on_state(state, reward, self)
                self._train_lock.release()

                # Increment episode count
                self._episode_ct += 1
                if (self._episode_ct >= self._max_episodes):
                    self._is_done = True

                for cb in callbacks:
                    cb(self)
                    
                # Intercept the loop and stop it
                if self._is_done:
                    self.stop()
                    return None
                
                try:
                    # Here goes the loop
                    action = post_process_action(action)
                    self._sock.emit('sim:action', action, self._ns.namespace)
                except Exception as e:
                    print('Emit error: ', e)
                    return None
            except Exception as e:
                print('Ns:', self._ns.namespace)
                print(f'An exception occured: {e}')
                self._is_done = True
                return None

        self._sock.on('sim:render', on_render, self._ns.namespace)
    
    def on_stop(self, on_stop):
        '''Attach custom simulation-loop stop callback'''
        self._on_stop = on_stop
    
    def stop(self, stop=True):
        '''Set stop flag to stop simulation-loop'''
        self._is_done = stop
        if self._on_stop:
            self._on_stop()

    def destroy(self):
        '''Destroy any sim created by this environment'''
        for sim_id in self._created_sims:
            requests.delete(f'{self.base_url}/sims/{sim_id}')
        self._created_sims = []