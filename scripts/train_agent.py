"""Train a DQN agent in the Amphitheater environment.

This script demonstrates how to create the environment and agent from the
`bems_drl` package and run a simple training loop.  Adjust the hyper‑
parameters and paths as needed.

"""
import os
from bems_drl import DQNAgent, AmphitheaterEnv


def main():
    # output directories
    out_dir = "./outputs"
    chkpt_dir = "./checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    # create environment
    env = AmphitheaterEnv(
        env_config={"output": out_dir},
        new_begin_month=10,
        new_end_month=10,
        new_begin_day=20,
        new_end_day=20,
        train=True,
        w_t=100,
        w_co2=10,
        w_dh=10,
        w_elc=1,
    )

    # instantiate agent
    agent = DQNAgent(
        gamma=0.99,
        epsilon=1.0,
        lr=1e-3,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=10_000,
        eps_min=0.05,
        batch_size=128,
        replace=384,
        eps_dec=0.9 * 100,
        chkpt_dir=chkpt_dir,
        algo="DQNAgent",
        env_name="amphitheater",
    )

    n_episodes = 100
    for episode in range(n_episodes):
        done = False
        obs, _ = env.reset()
        score = 0
        while not done:
            action = agent.choose_action_test(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_
            score += reward
            done = terminated or truncated
        agent.decrement_epsilon()
        print(f"Episode {episode+1}, score={score:.2f}, epsilon={agent.epsilon:.2f}")

    # save model
    agent.save_models()
    env.close()


if __name__ == "__main__":
    main()
