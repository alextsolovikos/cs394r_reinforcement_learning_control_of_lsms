import torch
import gym
import numpy as np
from scipy.interpolate import interp1d

class ObservationModel2D(torch.nn.Module):
    """
    PyTorch class for observing particles on a fixed grid.
    """
    def __init__(self, min_coord=[1.,0.], max_coord=[2., 1.], n_cells=[8, 4]):
        super().__init__()
        x = torch.linspace(min_coord[0], max_coord[0], n_cells[0]+1)
        y = torch.linspace(min_coord[1], max_coord[1], n_cells[1]+1)

        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        self.X, self.Y = X, Y
        
        self.nx, self.ny = n_cells
        
        self.min_coord = torch.stack((X[:-1,:-1], Y[:-1,:-1])).reshape(2,-1)
        self.max_coord = torch.stack((X[1:,1:], Y[1:,1:])).reshape(2,-1)
    
    def forward(self, particles):
        if not torch.is_tensor(particles):
            particles = torch.from_numpy(particles)
        left = particles.unsqueeze(-1) - self.min_coord.unsqueeze(0)
        right = self.max_coord.unsqueeze(0) - particles.unsqueeze(-1)
        return torch.sum(torch.all(torch.logical_and(left >= 0, right > 0) , dim=1),0).reshape(self.nx, self.ny).numpy()

# Define your problem using python and openAI's gym API:
class MultipleParticlesInFlowDelayedObsDiscrete(gym.Env):
    """Environment in which a particle moves at a constant speed in the x direction
    and at a varying speed in the y direction that depends on the input and the location.
    Action space: [0,1]
    State space: (x,y) \in R^2
    Reward: -y
    """

    def __init__(self, config):
        # Make the space (for actions and observations) configurable.
        self.action_space = gym.spaces.Discrete(2)
        # Since actions should repeat observations, their spaces must be the
        # same.
        self.N_max = config.get("hist_len_max", 10)
        self.N_min = config.get("hist_len_min", 0)
        self.n_obs = self.N_max - self.N_min
        
        self.obs_min = np.array(config.get("obs_min", [0.0, 0.0]))
        self.obs_max = np.array(config.get("obs_max", [8.0, 1.0]))
        self.n_particles = config.get("n_particles", 100)
        self.n_cells = config.get("n_cells", [1,1])
#         self.obs_model = ObservationModel2D(min_coord=self.obs_min.tolist(), max_coord=self.obs_max.tolist(), n_cells=[8,8])
        self.obs_model = ObservationModel2D(min_coord=self.obs_min.tolist(), max_coord=self.obs_max.tolist(), n_cells=self.n_cells)
        self.observation_space = gym.spaces.Box(low=np.zeros(self.n_obs*np.prod(self.n_cells)), high=np.ones(self.n_obs*np.prod(self.n_cells)), dtype=np.float64)
#         self.observation_space = gym.spaces.Box(low=np.zeros(2), high=self.obs_max, dtype=np.float64)
#         self.observation_space = gym.spaces.Box(low=np.zeros((8,8,1)), high=self.n_particles*np.ones((8,8,1)), dtype=np.float64)
#       self.observation_space = gym.spaces.Box(low=np.zeros(self.N_max-self.N_min), high=np.ones(self.N_max-self.N_min), dtype=np.float64)
#         self.state_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([15.0, 1.0]), dtype=np.float64)
        self.state_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([15.0, 1.0]), dtype=np.float64)
        self.cur_obs = None
        self.cur_state = None
        self.cur_action = 0.0
        self.cur_reward_downwash = None
        self.cur_reward_actuation = None
        self.cur_reward_total = None
        self.episode_len = 0
        self.dt = config.get("dt", 0.05)
        self.xc = config.get("xc", 5.0)
        self.yc = config.get("yc", 0.0)
        self.sigma_x = config.get("sigma_x", 0.5)
        self.sigma_y = config.get("sigma_y", 0.2)
        self.gain = config.get("gain", 0.5)
        self.act_cost_weight = config.get("act_cost_weight", 0.001)
        self.lamb = config.get("lamb", 1.0)

        # Turbulent boundary layer velocity profile
        tbl_data = np.loadtxt('tbl_velocity_profile.dat')
        self.u_tbl = interp1d(tbl_data[:,0], tbl_data[:,1], kind='cubic')
    
    def observe(self, state):
#         return self.obs_model(state).reshape(8,8,1)
        return self.obs_model(state).flatten().astype(np.float64) / self.n_particles
#         return state if np.all(np.logical_and((state - self.obs_min > 0), (self.obs_max - state > 0))) else np.zeros(2)
#         return np.array([])
#         return state.clip(env.obs_min, env.obs_max)

    def reset(self):
        """Resets the episode and returns the initial observation of the new one."""
        # Sample random volume of particles.
        self.cur_particles = np.random.multivariate_normal(
            mean=np.array([1.5,0.3]) + np.random.rand(2) * np.array([0.5,0.2]),
            cov = np.diag(
                np.array([0.15,0.06]) + np.random.rand(2) * np.array([0.6,0.1])
            )**2, size=self.n_particles
        )

        self.cur_particles = self.cur_particles[self.cur_particles[:,1] > 0]
        self.cur_particles = self.cur_particles[self.cur_particles[:,1] < 1]
        self.cur_particles = self.cur_particles[self.cur_particles[:,0] > 0]
        self.cur_state = np.mean(self.cur_particles, axis=0)
        
        # Reset the episode len.
        self.episode_len = 0
        # Sample a random number from our observation space.
#         self.cur_state = np.array([self.init_space.sample() for _ in range(self.n_particles)])
        self.cur_obs = self.observe(self.cur_particles)
        self.cur_obs_hist = [np.zeros(np.prod(self.n_cells)) for _ in range(self.N_max)]
        self.cur_obs_hist[0] = self.cur_obs
        self.cur_action = 0.0
        # Return initial observation.
        return np.hstack(self.cur_obs_hist)[self.N_min:]
    
    def jet_dist(self, x, y):
#         return np.exp(-(x - self.xc)**2/2./self.sigma_x**2) * np.exp(-(y - self.yc)**2/2./self.sigma_y**2)
        return np.maximum((x - self.xc)/self.sigma_x * np.exp(1-(x - self.xc)/self.sigma_x) * np.exp(-(y - self.yc)**2/2./self.sigma_y**2), 0.0)

    def step(self, action):
        """Takes a single step in the episode given `action`
        Returns: New observation, reward, done-flag, info-dict (empty).
        """
        
        self.cur_action = self.cur_action + self.lamb * (action - self.cur_action)
        self.episode_len += 1
        ux = self.u_tbl(self.cur_particles[:,1])
        uy = - self.cur_action * self.gain * self.jet_dist(self.cur_particles[:,0], self.cur_particles[:,1]) # if self.cur_state[1] > 0.05 else 0.0
        uy[self.cur_particles[:,1] < 0.05] = 0.0
        velocity = np.vstack([ux,uy]).T
        self.cur_particles = self.cur_particles + velocity * self.dt
        self.cur_obs = self.observe(self.cur_particles)
                
        self.cur_obs_hist.insert(0,self.cur_obs)
        self.cur_obs_hist.pop()
        
        self.cur_state = np.mean(self.cur_particles, axis=0)
        
        done = np.mean(self.cur_particles[:,0]) >= 14
#         reward = -self.cur_obs[1] if self.cur_obs[0] > self.xc else 0
#         reward = -self.cur_obs[1] if done else - 0.001 * action[0]
#         reward = -self.cur_state[1] if done else - self.act_cost_weight * action[0]
        self.cur_reward_downwash = -np.sum(uy)
        self.cur_reward_actuation = -action
        self.cur_reward_total = self.cur_reward_downwash + self.act_cost_weight * self.cur_reward_actuation
#         reward = -(uy-self.v_target)**2 - self.act_cost_weight * action[0]**2

        return np.hstack(self.cur_obs_hist)[self.N_min:], self.cur_reward_total, done, {}
