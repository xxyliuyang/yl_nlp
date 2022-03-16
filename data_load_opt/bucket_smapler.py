from torch.utils.data import Sampler
import random

class BucketSampler(Sampler):
    def __init__(self, data_source, padding_noise=0.1):
        super().__init__(data_source)
        self.lengths = [len(x) for x in data_source.features]
        self.padding_noise = padding_noise

    def _add_noise_to_value(self, value: int):
        noise_value = value * self.padding_noise
        noise = random.uniform(-noise_value, noise_value)
        return value + noise

    def __iter__(self):
        self.noisy_lengths = [self._add_noise_to_value(l) for l in self.lengths]
        indice_lengths = [(idx, length) for idx, length in enumerate(self.noisy_lengths)]
        indice_lengths.sort(key=lambda x: x[1])
        return iter([indice_length[0] for indice_length in indice_lengths])

    def __len__(self):
        return len(self.lengths)
