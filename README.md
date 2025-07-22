# DRL Agent for Building Energy Management System (BEMS)

This repository provides a deep reinforcement learning (DRL) framework for controlling an HVAC system in a building using a **Deep Q–Network (DQN)**.  It integrates with [EnergyPlus](https://energyplus.net/) via the [`rleplus`](https://github.com/Idiap/rllib-energyplus) API to simulate the building and HVAC dynamics and applies reinforcement learning to minimise energy consumption while maintaining occupant comfort.

## Features

- **Modular architecture** – The code is organised into a Python package (`bems_drl`) with clear separation between the neural network, the agent logic, the replay buffer and the environment definition.
- **Custom Gym environment** – The `AmphitheaterEnv` class in `bems_drl/env.py` wraps EnergyPlus into an OpenAI Gym–compatible environment.  It exposes state variables (temperatures, CO₂ concentration, occupancy, etc.), defines a discrete action space and implements a weighted reward function combining comfort and energy terms.
- **Deep Q–Network (DQN)** – The `DeepQNetwork` class implements a simple three‑layer fully connected network with adjustable hidden sizes.  It uses PyTorch for automatic differentiation and supports loading/saving checkpoints.
- **Experience replay** – `ReplayBuffer` stores past experiences to decorrelate training samples.
- **Agent implementation** – `DQNAgent` wraps the DQN network and replay buffer, implements epsilon‑greedy exploration, target network updates and the DDQN update rule.
- **Example script** – The `examples/main_dqn.py` script shows how to instantiate the environment and agent, run training episodes, save results and plot key metrics.

## Repository structure

```text
bems_drl_agent/                # repository root
├── bems_drl/                  # Python package containing core modules
│   ├── __init__.py            # package initialisation and exports
│   ├── deep_q_network.py      # neural network definition
│   ├── dqn_agent.py           
# DQN agent implementation
│   ├── replay_memory.py       # experience replay buffer
│   ├── env.py                 # custom EnergyPlus Gym environment (AmphitheaterEnv)
│   └── energyplus_runner.py   # low‑level runner interfacing with EnergyPlus (advanced)
├── examples/
│   └── main_dqn.py            # sample script demonstrating training and evaluation
├── scripts/                   # place for helper utilities or command‑line tools
├── requirements.txt           # Python dependencies
├── .gitignore                 # patterns for files to ignore in version control
└── README.md                  # this file
```

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd bems_drl_agent
   ```

2. **Install dependencies**  (preferably inside a virtual environment):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The key libraries are:

   - `numpy`, `pandas` and `matplotlib` for numerical operations and plotting
   - `scikit-learn` for feature scaling
   - `gymnasium` for the reinforcement learning API
   - `torch` for the neural network and automatic differentiation
   - `rleplus` and `energyplus` for the building simulation backend

3. **Download EnergyPlus model and weather files**

   The example environment (`AmphitheaterEnv`) expects an EnergyPlus IDF model and weather (EPW) file.  Copy these files into the `bems_drl` package directory (or adjust `env.py` accordingly).  The default names used are `model.idf` and `LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw`.

   You can obtain the model from your EnergyPlus installation or from the [rleplus examples directory](https://github.com/Idiap/rllib-energyplus/tree/main/examples/amphitheater).  The weather files are available from the [EnergyPlus weather database](https://energyplus.net/weather).

## Usage

The package can be used programmatically, or you can run the example script:

### Programmatic use

```python
from bems_drl import DQNAgent, AmphitheaterEnv

# create environment (adjust runperiod dates and reward weights as needed)
env = AmphitheaterEnv(
    env_config={"output": "./outputs"},
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
    chkpt_dir="./checkpoints",
    algo="DQNAgent",
    env_name="amphitheater"
)

# training loop (simplified)
for episode in range(100):
    done = False
    obs, _ = env.reset()
    score = 0
    while not done:
        action = agent.choose_action_test(obs)
        obs_, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(obs, action, reward, obs_, done)
        agent.learn()
        obs = obs_
        done = terminated or truncated
        score += reward
    agent.decrement_epsilon()
    print(f"episode={episode+1}, score={score:.2f}, epsilon={agent.epsilon:.2f}")

# save trained model
agent.save_models()
```

### Example script

The `examples/main_dqn.py` file reproduces the original training and evaluation routine supplied by the user.  It includes custom plotting and saves training metrics to disk.  To run it, make sure you have downloaded the required EnergyPlus model and weather files and then execute:

```bash
python examples/main_dqn.py
```

You may need to adjust file paths inside the script to match your local directory structure (e.g. for checkpoint directories and output files).

## Contributing

Contributions, bug reports and feature requests are welcome.  If you find a problem or have a suggestion, please open an issue or submit a pull request.

## License

This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.
