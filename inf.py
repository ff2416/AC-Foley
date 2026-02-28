import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from ac_foley.eval_utils import ModelConfig, generate, load_video, make_video
from ac_foley.model.flow_matching import FlowMatching
from ac_foley.model.networks import AC_Foley, get_ac_foley
from ac_foley.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

SAMPLE_RATE = 44100
NUM_TIMBRE_SAMPLES = 89088  # ~2 seconds at 44100 Hz


def load_audio(audio_path: str) -> torch.Tensor:
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(dim=0)  # stereo -> mono
    audio = audio / audio.abs().max() * 0.95  # normalize

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sr, SAMPLE_RATE,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method='sinc_interp_kaiser',
            beta=14.769656459379492,
        )
        audio = resampler(audio)

    if audio.size(0) < NUM_TIMBRE_SAMPLES:
        audio = torch.cat([audio, torch.zeros(NUM_TIMBRE_SAMPLES - audio.size(0))])
    else:
        audio = audio[:NUM_TIMBRE_SAMPLES]

    return audio


def process_video(
    video_path: Path,
    args,
    model: ModelConfig,
    net: AC_Foley,
    fm: FlowMatching,
    feature_utils: FeaturesUtils,
    device: str,
    dtype: torch.dtype,
    audio: torch.Tensor,
):
    log.info(f'Processing: {video_path}')

    audio_num_sample = audio.shape[0] if audio is not None else NUM_TIMBRE_SAMPLES

    video_info = load_video(video_path, args.duration)
    clip_frames = None if args.mask_away_clip else video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    duration = video_info.duration_sec

    model.seq_cfg.duration = duration
    model.seq_cfg.audio_num_sample = audio_num_sample
    net.update_seq_lengths(
        model.seq_cfg.latent_seq_len,
        model.seq_cfg.clip_seq_len,
        model.seq_cfg.sync_seq_len,
        model.seq_cfg.audio_seq_len,
    )

    log.info(f'Prompt: {args.prompt}')
    log.info(f'Negative prompt: {args.negative_prompt}')

    audios = generate(
        clip_frames,
        sync_frames,
        [args.prompt],
        audio,
        negative_text=[args.negative_prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=torch.Generator(device=device).manual_seed(args.seed),
        cfg_strength=args.cfg_strength,
    )

    out_audio = audios.float().cpu()[0]

    # build output stem: {video}_{audio}_{prompt}, skip empty parts
    def _safe(s: str) -> str:
        return s.replace('/', '_').replace(' ', '_')[:40]

    parts = [video_path.stem]
    if args.audio:
        parts.append(_safe(Path(args.audio).stem))
    if args.prompt:
        parts.append(_safe(args.prompt))
    stem = '_'.join(parts)

    args.output.mkdir(parents=True, exist_ok=True)
    wav_path = args.output / f'{stem}.wav'
    torchaudio.save(wav_path, out_audio, model.seq_cfg.sampling_rate)
    log.info(f'Audio saved to {wav_path}')

    if not args.skip_video_composite:
        video_save_path = args.output / f'{stem}.mp4'
        make_video(video_info, video_save_path, out_audio, sampling_rate=model.seq_cfg.sampling_rate)
        log.info(f'Video saved to {video_save_path}')


@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = ArgumentParser()
    parser.add_argument('--video', type=Path, required=True, help='Input video file (.mp4)')
    parser.add_argument('--audio', type=str, default='', help='Optional reference audio for timbre conditioning')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative text prompt')
    parser.add_argument('--duration', type=float, default=8.0, help='Duration in seconds')
    parser.add_argument('--cfg_strength', type=float, default=4.5, help='CFG strength')
    parser.add_argument('--num_steps', type=int, default=25, help='Number of diffusion steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (-1 for random)')
    parser.add_argument('--mask_away_clip', action='store_true', help='Disable CLIP visual conditioning')
    parser.add_argument('--skip_video_composite', action='store_true', help='Skip saving composited video')
    parser.add_argument('--full_precision', action='store_true', help='Use float32 instead of bfloat16')
    parser.add_argument('--output', type=Path, default='./results', help='Output directory')
    parser.add_argument('--model_path', type=str, default='./weights/ac_foley.pth', help='Model checkpoint path')
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = torch.seed()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        log.warning('CUDA not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    model = ModelConfig(
        model_path=Path(args.model_path),
        vae_path=Path('./weights/v1-44.pth'),
    )

    net: AC_Foley = get_ac_foley().to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=args.num_steps)
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        need_vae_encoder=True,
    ).to(device, dtype).eval()

    audio = load_audio(args.audio) if args.audio else None

    process_video(args.video, args, model, net, fm, feature_utils, device, dtype, audio)


if __name__ == '__main__':
    main()