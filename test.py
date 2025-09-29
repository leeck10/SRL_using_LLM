# coding=utf-8

__author__ = 'Hyunsun Hwang'
__version__ = '2025-9-30'

import argparse
import json
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TRANSFORMERS_VERBOSITY"]="error"

from tqdm import tqdm, trange

import util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_devices", default="0", type=str)
    parser.add_argument("--llm_name", default="google/gemma-2-27b-it", type=str)
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--test_file", default=None, type=str, required=True)
    parser.add_argument("--frame_file", default=None, type=str, required=True)
    parser.add_argument("--bert_name", default="klue/bert-base", type=str)
    parser.add_argument("--distance", default="euclidean", type=str)
    parser.add_argument("--TopK", default=0, type=int)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    
    model_id = args.llm_name
    train_filename = args.train_file
    test_filename = args.test_file
    frame_filename = args.frame_file
    bert_name = args.bert_name
    
    dist_method = args.distance
    #dist_method = "euclidean"
    #dist_method = "cosine"
    #dist_method = "mahalanobis"

    dist_feature = "last"
    #dist_feature = "avr_all"
    #dist_feature = "norm_all"
    
    search_k = args.TopK
    
    train_data = util.load_srl_data(train_filename, frame_filename)
    test_data = util.load_srl_data(test_filename, frame_filename)
    
    if len(train_data) < search_k :
        search_k = len(train_data)
    
    db = util.BertDB(bert_model_name=bert_name, use_gpu=True, metric=dist_method, extraction_method=dist_feature)
    
    # 데이터 추가
    for i, dd in tqdm(enumerate(train_data), desc=os.environ["CUDA_VISIBLE_DEVICES"]+"-add_item:", total=len(train_data), bar_format="{l_bar}{r_bar}") :
        db.add_item(dd["sentence_org"]+" [SEP] "+dd["v_org"]+" [SEP]", i)
        #print("input :", i)

    # 공분산 행렬 업데이트
    db.update_covariance_matrix()

    
    seleceted_order = list(range(search_k))
    
    pp1 = "다음은 한국어 의미역 결정의 부가격 의미역 정의이다.\n<ARGM-LOC 장소 (locatives)>\n<ARGM-DIR 방향 (directional)>\n<ARGM-CND 조건 (condition)>\n<ARGM-MNR 방법 (manner)>\n<ARGM-TMP 시간 (temporal)>\n<ARGM-EXT 범위 (extent)>\n<ARGM-PRD 보조 서술(secondary predication)>\n<ARGM-PRP 목적 (purpose clauses)>\n<ARGM-CAU 발생 이유 (cause clauses)>\n<ARGM-DIS 담화 연결 (discourse)>\n<ARGM-NEG 부정 (negation)>\n<ARGM-INS 도구 (instrument)>\n<ARGM-ADV 부사어 (adverbial)>\n<AUX 보조용언 (Auxiliary Verb)>\n"
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        #attn_implementation="flash_attention_2",
        #trust_remote_code=True,
    )
    
    data_iter = tqdm(enumerate(test_data), desc=os.environ["CUDA_VISIBLE_DEVICES"]+"-Evaluation:", total=len(test_data), bar_format="{l_bar}{r_bar}")
    for iii, tt in data_iter :
        query = tt["sentence_org"]+" [SEP] "+tt["v_org"]+" [SEP]" #case1
        loaded_results = db.search(query, k=search_k)
        ex = []
        for ddd in loaded_results :
            ex.append(train_data[ddd[1]])
        
        prompt_v = ""
        prompt_v = prompt_v + pp1 + "\n\n"
        for iid in seleceted_order :
            e = ex[iid]
            prompt_v = prompt_v + "Q: 다음 문장\n[" + e["sentence"] + "]\n에서 동사 [<predicate>" + e["v_org"] + "</predicate>]에 대한 의미역 결정 분석을 해서 conll 포맷으로 출력해줘. 이때 문장 분리를 하지말고 어절 그대로 사용하며, 가질 수 있는 필수격은\n" + e["roles"] + "\n이다. 이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것. 추가설명이나 반복은 하지 말 것.\nA: " + e["gold"] + tokenizer.eos_token + "\n\n"
        prompt_v = prompt_v + "Q: 다음 문장\n[" + tt["sentence"] + "]\n에서 동사 [<predicate>" + tt["v_org"] + "</predicate>]에 대한 의미역 결정 분석을 해서 conll 포맷으로 출력해줘. 이때 문장 분리를 하지말고 어절 그대로 사용하며, 가질 수 있는 필수격은\n" + tt["roles"] + "\n이다. 이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것. 추가설명이나 반복은 하지 말 것.\nA: "
        
        input_ids = tokenizer(prompt_v, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **input_ids,
            max_new_tokens=512,
            #do_sample=True,
            do_sample=False,
            #temperature=0.3,
            temperature=0.0,
            #top_p=0.9,
        )
        
        pred_result = tokenizer.decode(outputs[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        print("input prompt :", iii)
        print(prompt_v)
        print("")
        print("output")
        print(pred_result)
        print("")




if __name__ == "__main__":
    main()












