from typing import Iterator, Optional, Protocol, Sequence

import numpy as np

from d3rlpy.dataset import (
    EpisodeBase,
    ReplayBufferBase,
    TransitionMiniBatch,
    TransitionPickerProtocol,
)
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.types import GymEnv
from d3rlpy.metrics.utility import evaluate_qlearning_with_environment

WINDOW_SIZE = 1024


class EvaluatorProtocol(Protocol):
    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        """Computes metrics.

        Args:
            algo: Q-learning algorithm.
            dataset: ReplayBuffer.

        Returns:
            Computed metrics.
        """
        raise NotImplementedError
    

def make_batches(
    episode: EpisodeBase,
    window_size: int,
    transition_picker: TransitionPickerProtocol,
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, episode.transition_count)
        transitions = [
            transition_picker(episode, index)
            for index in range(head_index, last_index)
        ]
        batch = TransitionMiniBatch.from_transitions(transitions)
        yield batch
    

class AveragePatientEvaluator(EvaluatorProtocol):
    r"""Returns average value estimation.

    This metric suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.

    .. math::

        \mathbb{E}_{s_t \sim D} [ \max_a Q_\theta (s_t, a)]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        episodes = self._episodes if self._episodes else dataset.episodes
        total_values = []
        for episode in episodes:
            patient_values = []
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                values = algo.predict_value(batch.observations, actions)
                patient_values += values.tolist()
            total_values.append(np.mean(patient_values))
        
        return float(np.mean(total_values))