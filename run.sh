python3.10 test.py \
--cuda_devices 0 \
--llm_name google/gemma-2-9b-it \
--train_file samples/train_samples.txt \
--test_file samples/test_samples.txt \
--frame_file samples/framefiles_sample.json \
--bert_name klue/bert-base \
--distance euclidean \
--TopK 5 \










