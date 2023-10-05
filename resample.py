"""
这个文件是预处理的第一步, 主要功能是重采样到 44100 Hz

librosa库的主要功能是用于音频、音乐分析、处理的python工具包，它提供了一些常见的时频处理、特征提取、绘制声音图形等功能，例如：

- 读取和写入音频文件，支持多种格式，如wav、mp3等¹²³。
- 重采样、获取音频时长、过零率等基本的音频处理操作¹²³。
- 短时傅里叶变换（STFT）、梅尔频谱图（Mel-spectrogram）、色谱图（Chromagram）等时频分析方法¹²³。
- 音高（Pitch）、节拍（Beat）、节奏（Tempo）、音色（Timbre）等音乐特征提取¹²³。
- 波形图（Waveplot）、频谱图（Spectrogram）、色谱图（Chromagram）等声音图形绘制¹²³。

librosa库是一个非常强大的python语音信号处理的第三方库，它可以用于多种音频、音乐相关的应用，如语音识别、语音合成、声纹识别、音乐生成、音乐推荐等。如果你想要了解更多关于librosa库的原理和方法，你可以参考以下的网页：

- [librosa官方文档](^4^)：这是librosa库的官方文档，它详细地介绍了librosa库的安装、使用和示例。
- [librosa教程](^5^)：这是一个由librosa开发者之一Brian McFee制作的教程，它用Jupyter Notebook的形式展示了librosa库的各种功能和应用。
- [librosa教程 - 知乎]：这是一个由知乎用户Great编写的教程，它用中文的方式总结了librosa库的一些重要且常用的功能。

我希望这些信息对你有帮助。如果你有任何其他的问题或需求，请随时告诉我。😊

源: 与必应的对话， 2023/10/5
(1) librosa音频处理教程 - 知乎. https://zhuanlan.zhihu.com/p/509839164.
(2) librosa 语音库 （一）简介-CSDN博客. https://blog.csdn.net/chumingqian/article/details/123404790.
(3) Python音频处理包——Librosa的安装与使用 - CSDN博客. https://blog.csdn.net/qq_39516859/article/details/80333431.
(4) 音频特征提取——librosa工具包使用 - LeeLIn。 - 博客园. https://www.cnblogs.com/xingshansi/p/6816308.html.
(5) undefined. https://pypi.tuna.tsinghua.edu.cn/simple.
"""

import argparse
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import librosa
import numpy as np
from rich.progress import track
from scipy.io import wavfile


def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)


def trim_wav(wav, top_db=40):
    # 移除静音部分
    return librosa.effects.trim(wav, top_db=top_db)


def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav


def resample_wav(wav, sr, target_sr):
    """
    重采样
    """
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def save_wav_to_path(wav, save_path, sr):
    wavfile.write(
        save_path,
        sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )


def process(item):
    """
    处理单个音频文件
    """
    spkdir, wav_name, args = item
    # 获取说话者的名字, 也就是目录名
    speaker = spkdir.replace("\\", "/").split("/")[-1]

    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)

        # 处理文件
        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        wav = normalize_peak(wav)
        resampled_wav = resample_wav(wav, sr, args.sr2)

        # 响度均衡
        if not args.skip_loudnorm:
            resampled_wav /= np.max(np.abs(resampled_wav))

        # 保存文件
        save_path2 = os.path.join(args.out_dir2, speaker, wav_name)
        save_wav_to_path(resampled_wav, save_path2, args.sr2)


"""
def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)

    with ThreadPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
"""
# multi process


def process_all_speakers():
    """
    处理所有说话者
    """
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            # 遍历所有的说话者目录
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                # 仅处理 wav 文件
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                # 进度条可视化
                for _ in track(concurrent.futures.as_completed(futures), total=len(futures), description="resampling:"):
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 采样频率
    parser.add_argument("--sr2", type=int, default=44100, help="sampling rate")
    # 输入目录
    parser.add_argument("--in_dir", type=str, default="./dataset_raw", help="path to source dir")
    # 输出目录
    parser.add_argument("--out_dir2", type=str, default="./dataset/44k", help="path to target dir")
    # 是否跳过响度均衡
    parser.add_argument("--skip_loudnorm", action="store_true", help="Skip loudness matching if you have done it")
    args = parser.parse_args()

    print(f"CPU count: {cpu_count()}")
    # 这是所有的目录
    speakers = os.listdir(args.in_dir)
    process_all_speakers()
