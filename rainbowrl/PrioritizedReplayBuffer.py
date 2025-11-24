import numpy
import torch
from rltrees import SumTree
from collections import deque


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        image_size: int,
        steps: int,
        max_priority: float,
        alpha: float,
        epsilon: float,
    ):
        self.capacity = capacity
        self.steps = steps
        self.max_priority = max_priority
        self.alpha = alpha
        self.epsilon = epsilon

        self.states = numpy.zeros((capacity, image_size, image_size), dtype=numpy.uint8)
        self.actions = numpy.zeros((capacity, 1), dtype=numpy.int64)
        self.rewards = numpy.zeros((capacity, 1), dtype=numpy.float32)
        self.next_states = numpy.zeros(
            (capacity, image_size, image_size), dtype=numpy.uint8
        )
        self.terminations = numpy.zeros((capacity, 1), dtype=numpy.bool_)
        self.tree = SumTree(capacity)

        self.ptr = 0
        self.size = 0

        self.nstep = deque(maxlen=steps)

    def add(self, state, next_state, action, reward, termination):
        self.nstep.append((state, action, reward, next_state, termination))

        if len(self.nstep) < self.steps:
            return

        reward, next_state, termination = self.nstep_info()
        state = self.nstep[0][0]
        action = self.nstep[0][1]

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminations[self.ptr] = termination

        priority = self.max_priority**self.alpha
        self.tree.update(self.ptr, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if termination:
            self.nstep.clear()

    def nstep_info(self):
        discount = 1.0
        reward = 0.0
        for _, _, current_reward, next_state, termination in self.nstep:
            reward += discount * current_reward
            if termination:
                return reward, next_state, True
            discount *= self.gamma
        last_nstep = self.nstep[-1]
        return reward, last_nstep[3], last_nstep[4]

    def sample(self, batch_size):
        total = self.sum_tree.total()
        segment = total / batch_size

        indices = [
            self.sum_tree.retrieve(numpy.random.uniform(segment * i, segment * (i + 1)))
            for i in range(batch_size)
        ]

        priorities = numpy.array(
            [self.sum_tree.tree[idx + self.capacity - 1] for idx in indices]
        )

        probs = priorities / total
        weights = (self.size * probs) ** -self.beta
        weights /= weights.max()

        def t(x):
            return torch.from_numpy(x[indices]).to(self.device)

        return {
            "states": t(self.buffer_obs),
            "actions": t(self.buffer_actions).unsqueeze(1),
            "rewards": t(self.buffer_rewards).unsqueeze(1),
            "next_states": t(self.buffer_next_obs),
            "terminations": t(self.buffer_dones).unsqueeze(1),
            "weights": torch.from_numpy(weights).to(self.device).unsqueeze(1),
            "indices": indices,
        }

    def update_priorities(self, indices, priorities):
        priorities = (numpy.abs(priorities) + self.eps) ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())
        for idx, p in zip(indices, priorities):
            self.tree.update(idx, p)
