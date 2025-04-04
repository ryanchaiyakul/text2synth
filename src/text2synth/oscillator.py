import abc
import warnings

import torch

try:
    from torchaudio.prototype.functional import extend_pitch, oscillator_bank
except:
    print("Could not import extend_pitch, and oscillator_bank. Please install torch nightly.")
    raise


class Oscillator(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for differentiable oscillators.

    This class provides the foundation for generating common waveform types
    (Sinusoid, Triangle, Sawtooth, Square) with specified frequencies. 
    Subclasses should implement the `forward` method to define the specific 
    waveform generation logic.
    """

    def __init__(self, sample_rate: torch.Tensor | float):
        """
        Initialize the Oscillator with the given sample rate.

        Args:
            sample_rate (Tensor or float): The sample rate in Hz. 
                Should be a tensor of shape (1) or a float value.

        Raises:
            ValueError: If the sample rate is not a single value tensor.
        """
        super().__init__()
        if isinstance(sample_rate, torch.Tensor) and sample_rate.size != 1:
            raise ValueError(
                f"Sample rate must be a single value. Found: {sample_rate.shape} instead."
            )
        self.sample_rate = sample_rate

    @abc.abstractmethod
    def forward(self, frequencies: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform with specified frequencies.

        Args:
            frequencies (Tensor): Tensor of shape (batch, time), in Hz.

        Returns:
            Tensor: The resulting waveform with shape (batch, time).
        """
        pass


class SinOscillator(Oscillator):
    """
    Simple single sinusoid oscillator.
    """

    def forward(self, frequencies: torch.Tensor) -> torch.Tensor:
        invalid = torch.abs(frequencies) >= (nyquist := self.sample_rate / 2)
        if torch.any(invalid):
            warnings.warn(
                f"Some frequencies exceed Nyquist frequency ({nyquist} Hz). "
                "This will cause aliasing. Amplitudes are not zeroed."
            )

        # Phase increment per timestep
        phase_increments = 2 * torch.pi * frequencies / self.sample_rate
        phases = torch.cumsum(phase_increments, dim=-1) % (2 * torch.pi)
        return torch.sin(phases)


class SawtoothOscillator(Oscillator):
    """
    Simple single sawtooth oscillator with configurable additional pitches.
    """

    def __init__(self, sample_rate: float, num_pitches: int):
        """
        Args:
            sample_rate (float): The sample rate in Hz.
            num_pitches (int): The number of pitches (count).
        """
        super().__init__(sample_rate)
        self.num_pitches = num_pitches

    def forward(self, frequencies) -> torch.Tensor:
        mults = [-((-1) ** i) / (torch.pi * i)
                 for i in range(1, 1 + self.num_pitches)]
        total_freqs = extend_pitch(frequencies.unsqueeze(-1), self.num_pitches)
        amp = extend_pitch(torch.ones_like(frequencies).unsqueeze(-1), mults)
        return oscillator_bank(
            total_freqs, amp, sample_rate=self.sample_rate)


class TriangleOscillator(Oscillator):
    """
    Simple single triangle oscillator with configurable additional pitches.
    """

    def __init__(self, sample_rate: float, num_pitches: int):
        """
        Args:
            sample_rate (float): The sample rate in Hz.
            num_pitches (int): The number of pitches (count).
        """
        super().__init__(sample_rate)
        self.num_pitches = num_pitches

    def forward(self, frequencies) -> torch.Tensor:
        mults = [2.0 * i + 1.0 for i in range(self.num_pitches)]
        total_freqs = extend_pitch(frequencies.unsqueeze(-1), mults)
        c = 8 / (torch.pi**2)
        mults = [c * ((-1) ** i) / ((2.0 * i + 1.0) ** 2)
                 for i in range(self.num_pitches)]
        amp = extend_pitch(torch.ones_like(frequencies).unsqueeze(-1), mults)
        return oscillator_bank(
            total_freqs, amp, sample_rate=self.sample_rate)


class SquareOscillator(Oscillator):
    """
    Simple single square oscillator with configurable additional pitches.
    """

    def __init__(self, sample_rate: float, num_pitches: int):
        """
        Args:
            sample_rate (float): The sample rate in Hz.
            num_pitches (int): The number of pitches (count).
        """
        super().__init__(sample_rate)
        self.num_pitches = num_pitches

    def forward(self, frequencies) -> torch.Tensor:
        mults = [2.0 * i + 1.0 for i in range(self.num_pitches)]
        total_freqs = extend_pitch(frequencies.unsqueeze(-1), mults)
        mults = [4 / (torch.pi * (2.0 * i + 1.0))
                 for i in range(self.num_pitches)]
        amp = extend_pitch(torch.ones_like(frequencies).unsqueeze(-1), mults)
        return oscillator_bank(total_freqs, amp, sample_rate=self.sample_rate)
