# distilgpt2: 82MB, fastest, architecturally identical to GPT-2
python hessenberg_test.py --model distilgpt2 --save hessenberg_distilgpt2.json

# or full GPT-2 if you want the exact model from the paper
#python hessenberg_test.py --model gpt2 --save hessenberg_gpt2.json

# verbose to see per-layer detail
#python hessenberg_test.py --model distilgpt2 --verbose
