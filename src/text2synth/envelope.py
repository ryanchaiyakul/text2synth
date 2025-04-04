import torch

from typing import Dict


class AmpEnvelope(torch.nn.Module):
    """
    Simple differentiable ADSR amplifier envelope which operates on the entire audio sample.
    """

    def __init__(self, n_decay: int):
        """
        Args:
            n_decay (int): Degree of polynomial decay. 
        """
        super().__init__()
        self.n_decay = n_decay

    def get_envelope(self, x: torch.tensor, params: Dict[str, torch.Tensor], eps=1e-5):
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
        attack_curve = torch.clamp(x / (attack + eps), max=1.0)

        # hold accounted by (x - (attack + hold))

        # decay drops to sustain
        decay_curve = torch.clamp((x - (attack + hold)) / (decay + 1e-5), 0.0, 1.0)
        decay_curve = -decay_curve ** self.n_decay * (1.0 - sustain)

        decay_curve = torch.clamp((x - (attack + hold)) / (decay + eps), 0.0, 1.0)
        decay_curve = (torch.exp(-decay_curve * 2.0) - 1) * (1.0 - sustain) # Exponential decay (adjust the decay factor for control)

        # release drops to 0.0 linearly
        #release_curve = (x - (1.0 - release)) * sustain
        release_curve = -torch.clamp((x - (1.0 - release)) * sustain / (release), 0.0, 1.0)

        # Combine the phases
        envelope = attack_curve + decay_curve  + release_curve

        import matplotlib.pyplot as plt
        plt.plot(x[0], attack_curve.detach().numpy()[0], label='a')
        plt.plot(x[0], decay_curve.detach().numpy()[0], label='d')
        plt.plot(x[0], release_curve.detach().numpy()[0], label='r')
        plt.plot(x[0], envelope.detach().numpy()[0], label='envelope')
        plt.title("Amplifier Envelope")
        plt.legend()
        plt.show()

        # Normalize the envelope and apply to the audio
        return torch.clamp(envelope, min=0.0, max=1.0)

    def forward(self, audio: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            audio (Tensor): Tensor of shape (batch, time).
            params (dict): A dictionary containing 'adhsr'. The sum of adhr must be less than 1.0.
        Returns:
            Tensor: The modified audio with shape (batch, time).
        """
        x = torch.linspace(0, 1.0, audio.shape[1])[
            None, :].repeat(audio.shape[0], 1)
        return audio * self.get_envelope(x, params)


"""
# Initialize with sustain
        out = torch.full((num_frames,), float(sustain))

        # attack
        if num_a > 0:
            torch.linspace(0.0, 1.0, num_a + 1, out=out[: num_a + 1])

        # hold
        if num_h > 0:
            out[num_a : num_a + num_h + 1] = 1.0

        # decay
        if num_d > 0:
            # Compute: sustain + (1.0 - sustain) * (linspace[1, 0] ** n_decay)
            i = num_a + num_h
            decay = out[i : i + num_d + 1]
            torch.linspace(1.0, 0.0, num_d + 1, out=decay)
            decay **= self.n_decay
            decay *= 1.0 - sustain
            decay += sustain

        # release
        if num_r > 0:
            torch.linspace(sustain, 0, num_r + 1, out=out[-num_r - 1 :])

        return audio * out
"""
