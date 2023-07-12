import argparse
import os
from ossaudiodev import SNDCTL_SEQ_RESETSAMPLES

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

from utils import str2bool


def formatter_indictts(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs-22k", cols[0] + ".wav")
            text = cols[1].strip()
            speaker_name = cols[2].strip()
            #items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
            items.append(wav_file)
    return items

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training and evaluation script for vocoder model ')

    # dataset parameters
    parser.add_argument('--dataset_name', default='indictts', choices=['ljspeech', 'indictts', 'googletts'])
    parser.add_argument('--language', default='ta', choices=['en', 'ta', 'te', 'kn', 'ml', 'hi', 'mr', 'bn', 'gu', 'or', 'as', 'raj', 'mni' 'all'])
    parser.add_argument('--dataset_path', default='../../datasets/{}/{}', type=str)   
    parser.add_argument('--speaker', default='all') # eg. all, female, male
    parser.add_argument('--eval_split_size', default=10, type=int)

    # model parameters
    parser.add_argument('--model', default='hifigan', choices=['hifigan'])
    parser.add_argument('--seq_len', default=8192, type=int)
    parser.add_argument('--pad_short', default=2000, type=int)
    parser.add_argument('--use_noise_augment', default=True, type=str2bool)

    # training parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_eval', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_workers_eval', default=8, type=int)
    parser.add_argument('--lr_gen', default=0.0001, type=float)
    parser.add_argument('--lr_disc', default=0.0001, type=float)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)

    # training - logging parameters 
    parser.add_argument('--run_description', default='None', type=str)
    parser.add_argument('--output_path', default='output_vocoder', type=str)
    parser.add_argument('--test_delay_epochs', default=0, type=int)
    parser.add_argument('--print_step', default=100, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--save_step', default=10000, type=int)
    parser.add_argument('--save_n_checkpoints', default=1, type=int)
    parser.add_argument('--save_best_after', default=10000, type=int)
    parser.add_argument('--target_loss', default='loss_1')
    parser.add_argument('--print_eval', default=False, type=str2bool)
    parser.add_argument('--run_eval', default=True, type=str2bool)

    # distributed training parameters
    parser.add_argument('--port', default=54321, type=int)
    parser.add_argument('--continue_path', default="", type=str)
    parser.add_argument('--restore_path', default="", type=str)
    parser.add_argument('--group_id', default="", type=str)
    parser.add_argument('--use_ddp', default=True, type=bool)
    parser.add_argument('--rank', default=0, type=int)
    #parser.add_argument('--gpus', default='0', type=str)

    return parser
    

def main(args):

    config = HifiganConfig(
        audio=BaseAudioConfig(
            trim_db=60.0,
            mel_fmin=0.0,
            mel_fmax=8000,
            log_func="np.log",
            spec_gain=1.0,
            signal_norm=False,
        ),
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size_eval,
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers_eval,
        run_eval=args.run_eval,
        test_delay_epochs=args.test_delay_epochs,
        save_step=args.save_step,
        save_best_after=args.save_best_after,
        save_n_checkpoints=args.save_n_checkpoints,
        target_loss=args.target_loss,
        epochs=args.epochs,
        seq_len=args.seq_len,
        pad_short=args.pad_short,
        use_noise_augment=args.use_noise_augment,
        eval_split_size=args.eval_split_size,
        print_step=args.print_step,
        plot_step=args.plot_step,
        print_eval=args.print_eval,
        mixed_precision=args.mixed_precision,
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        data_path=args.dataset_path.format(args.language),
        #output_path=f'{args.output_path}/{args.language}_{args.model}',
        output_path=args.output_path,
        distributed_url=f'tcp://localhost:{args.port}',
        dashboard_logger='wandb',
        project_name='vocoder',
        run_name=f'{args.language}_{args.model}_{args.speaker}',
        run_description=args.run_description,
        wandb_entity='gokulkarthik'
    )

    ap = AudioProcessor(**config.audio.to_dict())

    if args.speaker == 'all':
        meta_file_train="metadata_train.csv"
        meta_file_val="metadata_test.csv"
    else:
        meta_file_train=f"metadata_train_{args.speaker}.csv"
        meta_file_val=f"metadata_test_{args.speaker}.csv"
    train_samples = formatter_indictts(config.data_path, meta_file_train)
    eval_samples = formatter_indictts(config.data_path, meta_file_val)

    model = GAN(config, ap)

    trainer = Trainer(
        TrainerArgs(continue_path=args.continue_path, restore_path=args.restore_path, use_ddp=args.use_ddp, rank=args.rank, group_id=args.group_id), 
        config, 
        config.output_path, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )
    trainer.fit()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    parser = get_arg_parser()
    args = parser.parse_args()

    args.dataset_path = args.dataset_path.format(args.dataset_name, args.language)
    #args.dataset_path += '/wavs-22k'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
