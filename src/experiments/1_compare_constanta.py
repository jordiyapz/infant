import multiprocessing as mp
from multiprocessing import Process
from time import sleep
import os

from infant import Agent, Environment
from infant.helper import create_responder

def subprocess():
    model_name = 'cache-v1'
    model_path = f'model/{model_name}.pkl'

    agent = Agent(alpha=5e-6, beta=1e-5, epsilon=.8)
    env = Environment() 
    env.create()

    env.reset()
    env.on_state(create_responder(agent, action_multiplier=100))
    env.connect(disconnect_on_done=True)
    env.init(max_episodes=1000)
    env.wait()

    agent.save(model_path)


def simple(i):
    try:
        print('Start ', i)
        sleep(2)
        print('Done: ', os.getpid())
        return 0
    except Exception as e:
        print(e)

if __name__ == '__main__':
    subprocesses = [mp.Process(target=simple, args=(i,), name='hello') 
                                  for i in range(2)]
    print(subprocesses)
    for subprocess in subprocesses:
        subprocess: Process = subprocess
        subprocess.start()
    print(subprocesses)
    # for subprocess in subprocesses:
    subprocesses[0].join()
    subprocesses[1].join()
    print(subprocesses)

    print('All Done')