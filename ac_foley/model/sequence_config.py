import dataclasses
import math

@dataclasses.dataclass
class SequenceConfig:
    # general
    duration: float

    # audio
    sampling_rate: int
    spectrogram_frame_rate: int
    audio_num_sample: int
    latent_downsample_rate: int = 2

    # visual
    clip_frame_rate: int = 8
    sync_frame_rate: int = 25
    sync_num_frames_per_segment: int = 16
    sync_step_size: int = 8
    sync_downsample_rate: int = 2

    @property
    def num_audio_frames(self) -> int:
        # we need an integer number of latents
        return self.latent_seq_len * self.spectrogram_frame_rate * self.latent_downsample_rate

    @property
    def latent_seq_len(self) -> int:
        return int(
            math.ceil(self.duration * self.sampling_rate / self.spectrogram_frame_rate /
                      self.latent_downsample_rate))

    @property
    def clip_seq_len(self) -> int:
        return int(self.duration * self.clip_frame_rate)

    @property
    def sync_seq_len(self) -> int:
        num_frames = self.duration * self.sync_frame_rate
        num_segments = (num_frames - self.sync_num_frames_per_segment) // self.sync_step_size + 1
        return int(num_segments * self.sync_num_frames_per_segment / self.sync_downsample_rate)

    @property
    def audio_seq_len(self) -> int:
        return math.ceil(self.audio_num_sample / self.sync_downsample_rate / self.spectrogram_frame_rate)


CONFIG_AC = SequenceConfig(duration=8.0, sampling_rate=44100, spectrogram_frame_rate=512, audio_num_sample=89088)

if __name__ == '__main__':
    assert CONFIG_AC.latent_seq_len == 345
    assert CONFIG_AC.clip_seq_len == 64
    assert CONFIG_AC.sync_seq_len == 192
    assert CONFIG_AC.audio_seq_len == 87
    assert CONFIG_AC.num_audio_frames == 353280

