# Full run (all 5 blocks, all topics in your KB)
#python ctx_algebra_ollama_suite.py

# Quick test on one topic, blocks A and B only
#python ctx_algebra_ollama_suite.py --blocks A,B --topics "Albert Einstein"

# All blocks, save output
#python ctx_algebra_ollama_suite.py --save my_results.json --verbose

# Different model if you have it
#python ctx_algebra_ollama_suite.py --model llama3.2:3b

#    --blocks A,B,E \

python ctx_algebra_ollama_suite.py \
    --blocks D \
    --topics "Albert Einstein,GPT-2 Model" \
    --save results_first_run.json \
    --kb my_knowledge_source_1_fix_factscore_inv.jsonl \
    --verbose

# Full run all 5 blocks
python ctx_algebra_ollama_suite_v2.py \
    --blocks A,B,C,D,E \
    --save results_v2.json \
    --verbose

# Quick: just the fixed Block D to verify embedding similarity works
#python ctx_algebra_ollama_suite_v2.py \
#    --blocks D \
#    --save results_block_d_v2.json

# Blocks that confirmed well last time, now with D fixed
#python ctx_algebra_ollama_suite_v2.py \
#    --blocks A,B,D,E \
#    --save results_v2.json
