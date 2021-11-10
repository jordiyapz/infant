import torch as T
from infant import Agent, Environment

def create_responder(agent: Agent, action_multiplier=20):
    def responder(observation: T.Tensor, reward: T.Tensor, env: Environment):
        """Get observation and returns action"""
        if reward > 0:
            print('Reward: ', reward)
        try:
            # Preprocess image
            image = observation / 255
            state = T.unsqueeze(image, dim=0)

            # Train the agent
            if env.prev_state != None:
                try:
                    agent.train(env.prev_state, reward, state, env._is_done)
                except Exception as e:
                    print('Training Error')
                    print(e)
            
            # Predict correct action
            action = agent.choose_action(state)
            action = T.squeeze(action) * action_multiplier

            env.prev_state = state
            
            return action
        except Exception as e:
            print(e)
            env.stop()
            return T.ones((3,))

    return responder