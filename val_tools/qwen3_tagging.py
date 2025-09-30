

import os
from pydub import AudioSegment
import dashscope
import math
from dashscope import MultiModalConversation
import fire
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 如果没有设置环境变量，可以手动设置 api_key = "sk-xxx"

# 切割音频文件的函数
def split_audio(input_file, segment_duration_ms=30000):
    """
    将音频文件按 segment_duration_ms（毫秒）进行切割，并返回切割后的音频文件路径列表
    """
    # 加载音频文件
    audio = AudioSegment.from_file(input_file)
    
    # 计算切割的音频段数
    duration_ms = len(audio)
    segments = math.ceil(duration_ms / segment_duration_ms)
    
    output_paths = []
    for i in range(segments):
        # 每段音频的开始和结束时间
        start_time = i * segment_duration_ms
        end_time = min((i + 1) * segment_duration_ms, duration_ms)
        
        # 切割音频
        segment = audio[start_time:end_time]
        
        # 保存每个切割后的音频文件
        output_path = f"{input_file}_segment_{i+1}.mp3"
        segment.export(output_path, format="mp3")
        
        # 保存切割后的音频路径
        output_paths.append(output_path)
    
    return output_paths

# 调用 DashScope API 进行识别
def recognize_audio_segments(audio_segments, model_name):
    """
    对切割后的音频段进行逐一识别，并返回识别的歌词
    """
    all_results = []
    
    for segment_path in audio_segments:
        # 调用 DashScope API 进行 ASR 识别
        if model_name == "Qwen3":
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"text": "识别这首歌的歌词，并给出句子级别的时间戳，中文汉字给出简体字"}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"audio": segment_path},
                    ]
                }
            ]
            response = dashscope.MultiModalConversation.call(
                api_key=API_KEY,
                model="qwen3-asr-flash",
                messages=messages,
                result_format="message",
                asr_options={
                    "enable_lid": True,
                    "enable_itn": False
                }
            )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"audio": segment_path}],
                }
            ]
            response = MultiModalConversation.call(model="qwen-audio-asr", messages=messages)
            
        # 提取识别结果并添加到 all_results 中
        # import pdb; pdb.set_trace()
        if response.status_code == 200:
            if len(response.output.choices[0].message.content) > 0:
                result = response.output.choices[0].message.content[0]['text']
                all_results.append(result)
        else:
            print(f"Error: {response.status_code}, {response}")
    
    # 合并所有段的识别结果
    return "\n".join(all_results)

# 主函数：分割音频，识别并合并结果
def process_audio_file(input_audio_path, model_name):
    # 1. 切割音频
    audio_segments = split_audio(input_audio_path, segment_duration_ms=179999)  # 2 分钟 = 120000 毫秒
    
    # 2. 识别切割后的音频文件
    lyrics = recognize_audio_segments(audio_segments, model_name=model_name)
    
    # 3. 删除临时切割的音频文件
    for segment in audio_segments:
        os.remove(segment)
    
    return lyrics

def process_qwen3_audio(input_audio_path):
    messages = [
        {
            "role": "user",
            "content": [{"audio": input_audio_path}],
        }
    ]
    response = MultiModalConversation.call(model="qwen-audio-asr", messages=messages)
    if response.status_code == 200:
        if len(response.output.choices[0].message.content) > 0:
            result = response.output.choices[0].message.content[0]['text']
            return result
    else:
        print(f"Error: {response.status_code}, {response}")
        return ""
        
from concurrent.futures import ThreadPoolExecutor, as_completed

def main(model_name="Qwen3-Audio", max_workers=2):
    song_paths = list(Path("data").glob("*.mp3"))
    if model_name == "Qwen3":
        output_path = "./lyrics_asr/lyrics_qwen3_asr_new.jsonl"
    elif model_name == "Qwen3-Audio":
        output_path = "./lyrics_asr/lyrics_qwen3_audio_asr.jsonl"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    results = []

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_song = {executor.submit(process_audio_file, str(song), model_name): str(song) for song in song_paths}

        for future in tqdm(as_completed(future_to_song), total=len(song_paths), desc="Processing songs"):
            song_path = future_to_song[future]
            try:
                lyrics = future.result()
            except Exception as e:
                print(f"Error processing {song_path}: {e}")
                lyrics = ""
            results.append({"path": song_path, "lyrics": lyrics})
            print(lyrics)

    # 全部处理完毕后统一写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for record in tqdm(results, desc="Writing results"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)

    
    

