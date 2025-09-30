import re
import os
import json
import argparse
from tqdm import tqdm

import numpy as np
from scipy import stats

from val_tools.songrep_val import calculate_wer_detail, extract_alphanumeric, split_mixed_enzh


class textWord:
    def __init__(self):
        self.patten = r'\[(.*?)\]\[(.*?)\](.*)'
        pass
    
    def get_all_text(self, lyric_norm_list):
        all_text = ""
        for lyric_norm in lyric_norm_list:
            matches = re.findall(self.patten, lyric_norm)[0]
            lyric_content = matches[2]
            all_text += lyric_content

        return all_text, 0
    
    def cal_wer(self, hyp_text_list, ref_text_list):
        if isinstance(ref_text_list, str):
            ref_text = ref_text_list.replace("\n", "")
        else:
            ref_text, _ = self.get_all_text(ref_text_list)
        if isinstance(hyp_text_list, str):
            hyp_text = hyp_text_list.replace("\n", "")
            wer = 0
        else:
            hyp_text, wer = self.get_all_text(hyp_text_list)

        ref_text = extract_alphanumeric(ref_text)
        ref_text = ref_text.lower()
        ref_text_list = split_mixed_enzh(ref_text)

        hyp_text = extract_alphanumeric(hyp_text)
        hyp_text = hyp_text.lower()
        hyp_text_list = split_mixed_enzh(hyp_text)
        er, _ = calculate_wer_detail(ref_text_list, hyp_text_list)
        return er, wer, ref_text_list, hyp_text_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-p", dest="hyp", default="lyrics_asr/lyrics_qwen3_asr_new.jsonl")
    parser.add_argument("-r", dest="ref", default="label_fix.jsonl")
    args = parser.parse_args()

    tw = textWord()

    hyp_dict = {}
    with open(args.hyp, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        item = json.loads(line)
        hyp_dict[os.path.split(item["path"])[-1].split(".")[0]] = item

    ref_dict = {}
    with open(args.ref, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        item = json.loads(line)
        ref_dict[item["idx"]] = item
    
    total_der = 0
    total_wer = 0
    total_wer_num = 0
    wer_list = []
    der_list = []
    for key in tqdm(list(ref_dict.keys())):
        one_ref = ref_dict[key]
        one_hyp = hyp_dict[key]
        wer, pre_wer, r, h = tw.cal_wer(one_hyp["lyrics"], one_ref["lyric_norm"])
        wer_list.append(wer)

    wer_array = np.array(wer_list)
    mean_wer = np.mean(wer_array)
    std_wer = np.std(wer_array, ddof=1)
    se = std_wer / np.sqrt(n)
    ci_low, ci_high = stats.norm.interval(0.95, loc=mean_wer, scale=se)

    print(f"wer: {mean_wer}, std: {std_wer}, 95% ci: ({ci_low}, {ci_high})")
