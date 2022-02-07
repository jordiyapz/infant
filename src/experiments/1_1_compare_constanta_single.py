import asyncio
import sys
import time
from dotenv import dotenv_values
from tqdm import tqdm

from infant import Agent, Environment
from infant.helper import create_responder


def get_(obj, key, default):
    if key in obj:
        return obj[key]
    return default


async def main():
    CONFIG = dotenv_values(".env.local")
    BASE_URL = get_(CONFIG, 'BASE_URL', "http://localhost:3000")
    STEP_PER_EP = int(get_(CONFIG, 'STEP_PER_EP', 100))

    epsilon = float(sys.argv[1]) if len(sys.argv) > 1 else 1
    v = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    model_name_fmt = lambda epsilon, v: f"agent_e{f'{epsilon:.3f}'.replace('.','')}_v{v}.pkl"
    model_path_fmt = lambda model_name: f'store/model/{model_name}'

    model_name = model_name_fmt(epsilon, v)
    model_path = model_path_fmt(model_name)

    print(f'Epsilon: {epsilon:.3f}')
    print('Model: ', model_name)

    agent = Agent(alpha=5e-6, beta=1e-5, epsilon=epsilon)
    try:
        agent.load(model_path)
    except FileNotFoundError:
        pass

    env = Environment(base_url=BASE_URL)

    try:
        env.create()
        env.connect(disconnect_on_done=True)
        pbar = tqdm(total=STEP_PER_EP)
        env.on_state(create_responder(agent, action_multiplier=200),
                     callbacks=[lambda env: pbar.update(1)])
        env.init(max_episodes=STEP_PER_EP)
        env.wait()
        pbar.close()

        print('Total reward: ', env.total_reward)
        print('Reward count: ', env.total_reward // 40)
        
        agent.save(model_path)
    except KeyboardInterrupt:
        print('Stopped')
        return None
    except Exception as e:
        print(e)
    finally:
        env.destroy()


if __name__ == '__main__':
    asyncio.run(main())
