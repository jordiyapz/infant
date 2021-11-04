import socketio
import requests
import torch
import urllib
from torchvision.io import decode_png

class SimNs(socketio.ClientNamespace):
    def on_connect(self):
        print('Connected')
    def on_disconnect(self):
        print('Disconnected')

class Environment:
    def __init__(self, sim_id:str=None, base_url='http://localhost:3000'):
        self.base_url = base_url
        self.sim_id = sim_id
        self._is_stop = False
        self._created_sims = []
        self._on_stop = None

        self._sock = socketio.Client()
        self.wait = self._sock.wait

        if sim_id:
            self._register_namespace()

        self._episode_ct = 0 
        self._max_episodes = None
    
    def _register_namespace(self):
        self._ns = SimNs(f'/sim-{self.sim_id}')
        self._sock.register_namespace(self._ns)

    def create(self):
        '''Create new simulation'''
        response = requests.post(f'{self.base_url}/sims').json()
        self.sim_id = response['id']
        self._created_sims.append(self.sim_id)
        self._register_namespace()
    
    def connect(self, close_on_stop=True):
        '''Connect to socket.io server'''
        self._sock.on('connect', lambda: print(f'Connected to {self._ns.namespace}'))
        self._sock.connect(self.base_url)

        if close_on_stop:
            self._on_stop = lambda: self.close()

    def init(self, max_episodes=None):
        '''Send initialization signal to simulation server. 
        In other words, init simulation-loop.'''
        self._is_stop = False
        self._max_episodes = max_episodes
        self._sock.emit('sim:init', namespace=self._ns.namespace)

    def close(self):
        '''Close socket.io connection'''
        self._sock.disconnect()

    def on_state(self, on_state):
        '''Attach simulation-loop
        
        Parameters:
            on_state: Function that takes `state`, `reward`, `env` and return `action`.
        
        '''
        
        def on_render(data_uri, reward):
            '''Thin wrapper to decode data_uri and send encoded actions.'''
            try:
                # Convert data_uri to state
                res = urllib.request.urlopen(data_uri)
                raw_image:bytes = res.file.read()
                tensor = torch.frombuffer(raw_image, dtype=torch.uint8)
                state = decode_png(tensor)[:3]  # iqnore last alpha channel as it always full

                # Calls user-defined on_state function
                actions:torch.Tensor = on_state(state, reward, self)

                # Increment episode count
                self._episode_ct += 1
                if (self._episode_ct >= self._max_episodes):
                    self._is_stop = True

                # Intercept the loop and stop it
                if self._is_stop:
                    self.stop()
                    return None

                # Here goes the loop
                action_dict = dict(zip(('x', 'y', 'down'), map(float, actions)))
                self._sock.emit('sim:action', action_dict, self._ns.namespace)

            except Exception as e:
                print(f'An exception occured: {e}')
                self._is_stop = True
                return None

        self._sock.on('sim:render', on_render, self._ns.namespace)
    
    def on_stop(self, on_stop):
        '''Attach custom simulation-loop stop callback'''
        self._on_stop = on_stop
    
    def stop(self, stop=True):
        '''Set stop flag to stop simulation-loop'''
        self._is_stop = stop
        if self._on_stop:
            self._on_stop()

    def destroy(self):
        '''Destroy any sim created by this environment'''
        for sim_id in self._created_sims:
            requests.delete(f'{self.base_url}/sims/{sim_id}')
        self._created_sims = []