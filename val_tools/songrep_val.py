import argparse
import re
import json
import subprocess
from datetime import timedelta
from collections import Counter

import torchaudio
import torch
import numpy as np
from tqdm import tqdm
from megatron.tokenizer import build_tokenizer
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
from transformers import AutoModelForCausalLM

from mucodec.generate_1rvq import Tango
from run import megaInf
from val_tools.audio import load_track

class DiarizationIdErrorRate(DiarizationErrorRate):
    def compute_components(self, reference, hypothesis, uem=None, **kwargs):
        reference, hypothesis, uem = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap,
            returns_uem=True)
        return super(DiarizationErrorRate, self) \
            .compute_components(reference, hypothesis, uem=uem, skip_overlap=False, **kwargs)

class textStructure:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda")
        else:
            self.device = torch.device("cpu")
        self.der = DiarizationIdErrorRate(collar=0.5)

    def get_segment(self, struct_with_time):
        prediction = Annotation()
        for start, end, label in struct_with_time:
            prediction[Segment(start, end)] = label
        
        return prediction
    
    def cal_der(self, hyp_list, ref_list):
        reference = self.get_segment(ref_list)
        prediction = self.get_segment(hyp_list)
        der = self.der(reference, prediction) 
        return der


def cut_audio(input_path, out_path, start, end):
    if end < start:
        cmd = ["ffmpeg", "-i", input_path, "-ss", str(start), "-y", out_path]
    else:
        cmd = ["ffmpeg", "-i", input_path, "-ss", str(start), "-t", str(end - start), "-y", out_path]
    subprocess.check_output(cmd, stderr=subprocess.DEVNULL)


def find_most_frequent_element(arr):
    counter = Counter(arr)
    most_common = counter.most_common(1)
    try:
        most_frequent_element = most_common[0][0]
    except IndexError:
        most_frequent_element = ""
    return most_frequent_element


def extract_timestamp_text(lyric):
    pattern = r"\[(\d{2}):(\d{2})\.(\d+)\](.*)"
    matches = re.match(pattern, lyric)
    if matches:
        minutes = int(matches.group(1))
        seconds = int(matches.group(2))
        microsecond = int(matches.group(3))
        content = matches.group(4)
        timestamp = int(timedelta(minutes=minutes, seconds=seconds).total_seconds())
        timestamp = float(f"{timestamp}.{microsecond}")
    else:
        timestamp = -1.0
        content = ""

    return timestamp, content


def extract_second_text(lyric):
    matches = re.findall(r'\[(.*?)\](.*)', lyric)
    if matches:
        start, end = matches[0][0].split(':')
        start = float(start)
        end = float(end)
        content = matches[0][1]
    else:
        start, end = -1, -1
        content = ""
    return start, end, content


def get_time_content_pair(lyrics):
    time_content_pairs = []
    if len(lyrics) == 0:
        return time_content_pairs
    elif len(lyrics) == 1:
        timestamp, content = extract_timestamp_text(lyrics[0])
        time_content_pairs = [[[timestamp, -1], content]]
        return time_content_pairs
    for idx in range(len(lyrics) - 1):
        timestamp, content = extract_timestamp_text(lyrics[idx])
        next_timestamep, _ = extract_timestamp_text(lyrics[idx + 1])
        time_content_pairs.append([[timestamp, next_timestamep], content])
    timestamp, content = extract_timestamp_text(lyrics[idx + 1])
    time_content_pairs.append([[timestamp, -1], content])
    return time_content_pairs


def batch_time_content_pair(lyric_path, min_gap=30, offset=0):
    with open(lyric_path, "r") as fp:
        jd = json.load(fp)
    time_content_pairs = []
    times = []
    sentences = ""
    lyrics = jd.get("lyrics", [])
    for lyric in lyrics:
        timestamp, content = extract_timestamp_text(lyric)
        if len(times) == 0:
            times.append(timestamp - offset)
            sentences = content
        else:
            if len(times) == 1:
                times.append(timestamp + offset)
            else:
                times[1] = timestamp + offset
            if times[1] - times[0] > min_gap:
                time_content_pairs.append([times, sentences])
                times = [times[1] - offset]
                sentences = content
            else:
                sentences += content
    if len(times) == 1:
        times.append(-1)
        time_content_pairs.append([times, sentences])
    elif len(times) == 2:
        times[1] = -1
        time_content_pairs.append([times, sentences])
    
    return time_content_pairs


def calculate_cer_detail(ref, hyp):
    ref_chars = list(ref)
    hyp_chars = list(hyp)

    # 创建一个二维数组来保存编辑距离和操作
    dp = [[(0, '')] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    # 初始化第一行和第一列
    for i in range(len(ref_chars) + 1):
        dp[i][0] = (i, 'D')
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = (j, 'I')

    # 计算编辑距离和操作
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], 'M')
            else:
                replace_cost = dp[i - 1][j - 1][0] + 1
                insert_cost = dp[i][j - 1][0] + 1
                delete_cost = dp[i - 1][j][0] + 1
                min_cost = min(replace_cost, insert_cost, delete_cost)
                if min_cost == replace_cost:
                    dp[i][j] = (replace_cost, 'R')
                elif min_cost == insert_cost:
                    dp[i][j] = (insert_cost, 'I')
                else:
                    dp[i][j] = (delete_cost, 'D')

    # 统计插入、删除和替换的字符及其位置
    insertions = {}
    deletions = {}
    replacements = {}
    i = len(ref_chars)
    j = len(hyp_chars)
    while i > 0 or j > 0:
        if dp[i][j][1] == 'I':
            insertions[i - 1] = hyp_chars[j - 1]
            j -= 1
        elif dp[i][j][1] == 'D':
            deletions[i - 1] = ref_chars[i - 1]
            i -= 1
        elif dp[i][j][1] == 'R':
            replacements[i - 1] = (ref_chars[i - 1], hyp_chars[j - 1])
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1

    # 计算字错误率
    if len(ref_chars) > 0:
        cer = dp[len(ref_chars)][len(hyp_chars)][0] / len(ref_chars)
    else:
        cer = 100

    return cer, [insertions, deletions, replacements]


def calculate_wer_detail(reference_words, hypothesis_words):
    reference_words = [r.lower() for r in reference_words]
    hypothesis_words = [h.lower() for h in hypothesis_words]

    # 创建一个二维数组来存储编辑距离和操作
    dp = [[(0, '')] * (len(hypothesis_words) + 1) for _ in range(len(reference_words) + 1)]

    # 初始化第一行和第一列
    for i in range(len(reference_words) + 1):
        dp[i][0] = (i, 'D')
    for j in range(len(hypothesis_words) + 1):
        dp[0][j] = (j, 'I')

    # 计算编辑距离和操作
    for i in range(1, len(reference_words) + 1):
        for j in range(1, len(hypothesis_words) + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], 'C')
            else:
                deletion = dp[i - 1][j][0] + 1
                insertion = dp[i][j - 1][0] + 1
                substitution = dp[i - 1][j - 1][0] + 1
                min_cost = min(deletion, insertion, substitution)
                if min_cost == deletion:
                    dp[i][j] = (deletion, 'D')
                elif min_cost == insertion:
                    dp[i][j] = (insertion, 'I')
                else:
                    dp[i][j] = (substitution, 'S')

    # 统计插入、删除和替换的字符及其位置
    insertions = {}
    deletions = {}
    substitutions = {}
    i = len(reference_words)
    j = len(hypothesis_words)
    while i > 0 or j > 0:
        if dp[i][j][1] == 'I':
            if i in insertions:
                insertions[i].append(hypothesis_words[j - 1])
            else:
                insertions[i] = [hypothesis_words[j - 1]]
            j -= 1
        elif dp[i][j][1] == 'D':
            deletions[i] =  reference_words[i - 1]
            i -= 1
        elif dp[i][j][1] == 'S':
            substitutions[i] = (reference_words[i - 1], hypothesis_words[j - 1])
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1

    # 计算WER
    if len(reference_words) > 0:
        wer = dp[len(reference_words)][len(hypothesis_words)][0] / len(reference_words)
    else:
        wer = 100
    return wer, [insertions, deletions, substitutions]


def replace_chinese_place(input_str):
    chinese_pattern = re.compile(r'([\u4e00-\u9fff]+)\s+')
    out_str = re.sub(chinese_pattern, r'\1.', input_str)
    return out_str


def expand_with_tail(input_str, end_mark="."):
    split_idx = []
    delay = 0
    input_str_list = split_mixed_enzh(input_str)
    strip_str_list = []
    for idx, word in enumerate(input_str_list):
        if word == end_mark:
            split_idx.append(idx - 1 - delay)
            delay = delay + 1
            continue
        strip_str_list.append(word)
    return strip_str_list, split_idx


def extract_alphanumeric(input_str):
    pattern = r"[^'\u4e00-\u9fa5a-zA-Z\s0-9]"
    alp_txt = re.sub(pattern, "", input_str)
    return alp_txt.strip()


def split_mixed_enzh(text):
    words = re.findall(r"[a-zA-Z']+|\S", text)
    return words


def fix_and_add_tail(input_str_list, wer_detail, split_idx):
    insertions, deletions, _ = wer_detail
    # 对齐wer的序号，从1开始
    input_str_list = [""] + input_str_list
    fix_txt_list = []
    idx = 0
    for idx in range(len(input_str_list)):
        if idx in deletions:
            continue
        fix_txt_list.append(input_str_list[idx])
        if idx in insertions:
            # 倒序插入，需要反转
            insertion_list = insertions[idx][::-1]
            tmp_str = insertion_list[0]
            for ins_idx, ins_str in enumerate(insertion_list[1:]):
                # 英文之间引入空格
                if re.match(r"^[a-zA-Z'\s]+$", insertion_list[ins_idx]) and re.match(r"^[a-zA-Z'\s]+$", insertion_list[ins_idx-1]):
                    tmp_str = tmp_str + " " + ins_str
                else:
                    tmp_str += ins_str
            fix_txt_list.append(tmp_str)
        # 加入断点，对齐原序号
        if idx - 1 in split_idx:
            fix_txt_list.append(".")
    if len(fix_txt_list) > 0:
        fix_txt = fix_txt_list[0]
        for idx in range(1, len(fix_txt_list)):
            if re.match(r"^[a-zA-Z'\s]+$", fix_txt_list[idx]) and re.match(r"^[a-zA-Z'\s]+$", fix_txt_list[idx-1]):
                fix_txt = fix_txt + " " + fix_txt_list[idx]
            else:
                fix_txt += fix_txt_list[idx]
    else:
        fix_txt = ""
    return fix_txt

def escape_special_characters(text):
    special_characters = r'[\^$|?*+(){}\\]'
    text = re.sub(special_characters, r'\\\g<0>', text)
    text = text.replace(" ", r'\ ')
    return text


def cal_iou(segment_a, segment_b):
    x1, y1 = segment_a
    x2, y2 = segment_b

    min_a = min(x1, y1)
    max_a = max(x1, y1)
    min_b = min(x2, y2)
    max_b = max(x2, y2)

    intersection = max(0, min(max_a, max_b) - max(min_a, min_b))
    union = min((y1 - x1), (y2 - x2))
    if union > 0:
        iou = intersection / union
    else:
        iou = 0
    return iou

def check_language_by_text(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_count = len(re.findall(chinese_pattern, text))
    english_count = len(re.findall(english_pattern, text))
    chinese_ratio = chinese_count / len(text)
    english_ratio = english_count / len(text)
    if chinese_ratio >= 0.2:
        return "zh"
    elif english_ratio >= 0.5:
        return "en"
    else:
        return "unknow"


def parse_lyric(lyric):
    pattern = r'\[(.*)\]\[(\d+\.\d+):(\d+\.\d+)\](.*)'
    match = re.search(pattern, lyric)
    if match is None:
        pattern = r'\[(.*)\]\[(\d+):(\d+\.\d+)\](.*)'
        match = re.search(pattern, lyric)
    if match is None:
        pattern = r'\[(.*)\]\[(\d+\.\d+):(\d+)\](.*)'   
        match = re.search(pattern, lyric)     
    if match is None:
        pattern = r'\[(.*)\]\[(\d+):(\d+)\](.*)'   
        match = re.search(pattern, lyric)

    structure = match.group(1)
    start_time = float(match.group(2))
    end_time = float(match.group(3))
    content = match.group(4)

    return start_time, end_time, structure, content


def parse_output(text):
    try:
        match = re.search(r"\[(.*)\]\[(\d+\.\d+):(\d+\.\d+)\](.*)", text)
        start_time = float(match.group(2))
        end_time = float(match.group(3))
        structure = match.group(1)
        content = match.group(4)
    except:
        match = re.search(r"\[(.*)\]\[(\d+\.\d+):(\d+\.\d+)\]", text)
        start_time = float(match.group(2))
        end_time = float(match.group(3))
        structure = match.group(1)
        content = ""

    return start_time, end_time, structure, content

def repl(m):
    prev_char = m.group(1)
    dot = m.group(2)
    next_char = m.group(3)
    if (prev_char.isalpha() and prev_char.isascii()) or (next_char.isalpha() and next_char.isascii()):
        return prev_char + " " + next_char
    else:
        return prev_char + next_char
    
def repl_text(text):
    pattern = re.compile(r"(.)(\.)(.)")
    text = pattern.sub(repl, text)
    text = text.replace(".", "")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-q", dest="qwen_ckpt", default="SongPrep-7B/")
    parser.add_argument("-s", dest="ssl_ckpt", default="SongPrep-7B/muencoder.pt")
    parser.add_argument("-c", dest="codec_ckpt", default="SongPrep-7B/mucodec.safetensors")
    parser.add_argument("-d", dest="src_dir", default="data")
    parser.add_argument("-r", dest="ref_jsonl", default="label_fix.jsonl")
    args = parser.parse_args()

    vocal_file = "conf/vocab_type.yaml"
    model_path = args.qwen_ckpt
    ssl_path = args.ssl_ckpt
    codec_path = args.codec_ckpt
    src_dir = args.src_dir
    json_path = args.ref_jsonl

    tango = Tango(model_path=codec_path, ssl_path=ssl_path)
    maga = megaInf(model_path, vocal_file)
    tst = textStructure()

    with open(json_path, "r") as fp:
        lines = fp.readlines()

    total_wer = 0
    total_der = 0
    count_num = 0
    new_items = []
    for line in tqdm(lines):
        end_time = 0
        data = json.loads(line)
        lyrics = data["lyric_norm"]
        item = {"idx": data["idx"]}
        ref_content = []
        ref_structs = []
        for seg in range(len(lyrics)):
            try:
                this_start_time, this_end_time, this_structure, this_content = parse_lyric(lyrics[seg])
            except:
                print(lyrics[seg])
            if this_end_time > 300:
                break
            ref_content.append(this_content)
            ref_structs.append([this_start_time, this_end_time, this_structure])
            end_time = this_end_time

        src_wave, fs = load_track(f"{src_dir}/{data['idx']}.mp3", None, None)
        if (fs != 48000):
            src_wave = torchaudio.functional.resample(src_wave, fs, 48000)
        code = tango.sound2code(src_wave)[0][0][:int((end_time*25))] 
        # audio =  kaldiio.load_mat(ark_dict[data["idx"]])[0][0]
        pre_lyric = maga.run(code.cpu())

        ref_text = ".".join(ref_content)
        hyp_text = pre_lyric.replace("<|extra_1|>", "").replace("<|endoftext|>", "").replace("<|im_end|>", "")

        hyp_text_list = []
        hyp_structs = []
        for text in hyp_text.split(";"):
            try:
                start_time, end_time, structure, content = parse_output(text.strip())
            except:
                print(text)
                continue
            if content != "":
                hyp_text_list.append(content)
            hyp_structs.append([start_time, end_time, structure])
        item["lyric_norm"] = hyp_text.split(";")
        item["structure"] = hyp_structs

        hyp_text = ".".join(hyp_text_list)
        ref_text = repl_text(ref_text)
        ref_text = ref_text.lower()
        ref_text_list = split_mixed_enzh(ref_text.strip())

        hyp_text = repl_text(hyp_text)
        hyp_text = hyp_text.lower()
        hyp_text_list = split_mixed_enzh(hyp_text.strip())
        er, _ = calculate_wer_detail(ref_text_list, hyp_text_list)
        der = tst.cal_der(hyp_structs, ref_structs)
        # print(f"{data['idx']} WER: {er}, DER: {der}")
        max_words = max(Counter(hyp_text_list).values())
        if max_words < 100:
            total_wer += er
            total_der += der
            count_num += 1
            new_items.append(item)
        else:
            print(f"skip {data['idx']} with {max_words} max words")
            print(hyp_text)
            print(ref_text)

    print(f"WER on {count_num} items", total_wer / count_num)
    print(f"DER on {count_num} items", total_der / count_num)
