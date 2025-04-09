import torch
import math

from typing import Dict, Tuple
import matplotlib.pyplot as plt


class AmpEnvelope(torch.nn.Module):
    """
    Simple differentiable ADSR amplifier envelope which operates on the entire audio sample.
    """

    def __init__(self, n_decay: int):
        """
        Args:
            n_decay (int): Scale of exponential decay. 
        """
        super().__init__()
        self.n_decay = n_decay
        self.decay_scale = 1.0 - math.exp(-2.0)

    def get_params_dict(self, a: float, h: float, d: float, s: float, r: float, requires_grad=False) -> Dict[str, torch.Tensor]:
        """
        Convert float parameters into a tensor dictionary.

        Args:
            a (float): Attack duration [0,1].
            h (float): Hold duration [0,1].
            d (float): Decay duration [0,1].
            s (float): Sustain level [0,1].
            r (float): Release duration [0,1].
        """
        return {
            'a': torch.tensor([a], requires_grad=requires_grad),
            'h': torch.tensor([h], requires_grad=requires_grad),
            'd': torch.tensor([d], requires_grad=requires_grad),
            's': torch.tensor([s], requires_grad=requires_grad),
            'r': torch.tensor([r], requires_grad=requires_grad)
        }

    def get_envelope(self, t: torch.tensor, params: Dict[str, torch.Tensor], eps=1e-5):
        """
        Args:
            t (Tensor): Interpolation time tensor of shape (batch, time_step).
            params (dict): A dictionary containing 'adhsr'. The sum of adhr must be less than 1.0 and s <= 1.0.
            eps (float): Divide by 0 offset.
        Return:
            Tensor: Envelopes of shape (batch, time_step).
        """
        try:
            attack = params['a']
            hold = params['h']
            decay = params['d']
            sustain = params['s']
            release = params['r']
        except KeyError:
            raise ValueError("params is missing a key")

        # Ensure the parameters are within valid ranges
        assert 0 <= attack <= 1, f"Invalid attack: {attack}"
        assert 0 <= hold <= 1, f"Invalid hold: {hold}"
        assert 0 <= decay <= 1, f"Invalid decay: {decay}"
        assert 0 <= sustain <= 1, f"Invalid sustain: {sustain}"
        assert 0 <= release <= 1, f"Invalid release: {release}"

        if attack + hold + decay + release > 1.0:
            raise ValueError(
                "The sum of attack, hold, decay, and release cannot exceed 1.")

        # attack rises to 1.0 linearly
        attack_curve = torch.clamp(t / (attack + eps), max=1.0)

        # hold accounted by (x - (attack + hold))

        # decay drops to sustain
        delay_coeff = torch.clamp(
            (t - (attack + hold)) / (decay + eps), 0.0, 1.0)
        decay_curve = (torch.exp(-delay_coeff * 2.0) - 1.0) / \
            (self.decay_scale) * (1 - sustain)

        # sustain accounted by (1.0 - release)

        # release drops to 0.0 linearly
        release_curve = - \
            torch.clamp((t - (1.0 - release)), 0.0, 1.0) * \
            sustain / (release + eps)

        return torch.clamp(attack_curve + decay_curve + release_curve, min=0.0, max=1.0)

    def forward(self, audio: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            audio (Tensor): Tensor of shape (batch, time).
            params (dict): A dictionary containing 'adhsr'. The sum of adhr must be less than 1.0 and s <= 1.0.
        Returns:
            Tensor: The modified audio with shape (batch, time).
        """
        t = torch.linspace(0, 1.0, audio.shape[1])[
            None, :].repeat(audio.shape[0], 1)
        return audio * self.get_envelope(t, params)


class AmpEnvelopeNN(torch.nn.Module):

    """
    NN differentiable ADSR amplifier envelope which operates on the entire audio sample.
    """

    def __init__(self, steps: int, hidden_dim: Tuple[int, int] = (100, 200)):
        """
        Args:
            steps (int): Number of discrete steps within the envelope.
            hidden_dim (Tuple[int, int]): Hidden layer one and two's size.
        """
        super().__init__()
        self.steps = steps
        self._NN = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_dim[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[0], hidden_dim[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[1], steps),
            torch.nn.Sigmoid()
        )

    def get_params_dict(self, a: float, h: float, d: float, s: float, r: float, requires_grad=False) -> Dict[str, torch.Tensor]:
        """
        Convert float parameters into a tensor dictionary.

        Args:
            a (float): Attack duration [0,1].
            h (float): Hold duration [0,1].
            d (float): Decay duration [0,1].
            s (float): Sustain level [0,1].
            r (float): Release duration [0,1].
        """
        return {
            'a': torch.tensor([a], requires_grad=requires_grad),
            'h': torch.tensor([h], requires_grad=requires_grad),
            'd': torch.tensor([d], requires_grad=requires_grad),
            's': torch.tensor([s], requires_grad=requires_grad),
            'r': torch.tensor([r], requires_grad=requires_grad)
        }

    def get_envelope(self, t: torch.tensor, params: Dict[str, torch.Tensor], eps=1e-5):
        """
        Only the number of time_steps are required, but t is left for backwards compatability.

        Args:
            t (Tensor): Interpolation time tensor of shape (batch, time_step).
            params (dict): A dictionary containing 'adhsr'. The sum of adhr must be less than 1.0 and s <= 1.0.
            eps (float): Divide by 0 offset.
        Return:
            Tensor: Envelopes of shape (batch, time_step).
        """
        try:
            attack = params['a']
            hold = params['h']
            decay = params['d']
            sustain = params['s']
            release = params['r']
        except KeyError:
            raise ValueError("params is missing a key")
        x = torch.cat((attack, hold, decay, sustain, release), dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        env = self._NN(x)
        envelope = torch.nn.functional.interpolate(
            env.unsqueeze(0), size=t.shape[1], mode="linear"
        ).squeeze(0)

        return torch.clamp(envelope, 0.0, 1.0)

    def forward(self, audio: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            audio (Tensor): Tensor of shape (batch, time).
            params (dict): A dictionary containing 'adhsr'. The sum of adhr must be less than 1.0 and s <= 1.0.
        Returns:
            Tensor: The modified audio with shape (batch, time).
        """
        t = torch.linspace(0, 1.0, audio.shape[1])[
            None, :].repeat(audio.shape[0], 1)
        return audio * self.get_envelope(t, params)
