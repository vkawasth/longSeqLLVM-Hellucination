import ollama

# We use the 1B variant to closely match GPT-2's lean parametric behavior
response = ollama.generate(
    model='llama3.2:1b', 
    prompt="Your experiment baseline prompt here...",
    options={
        'num_ctx': 1024,      # Locks the window to GPT-2 context size
        'temperature': 0.7,   # Matches standard GPT-2 generation variance
        'top_k': 40,          # GPT-2's original default top_k filtering
        'top_p': 0.9          # GPT-2's original default nucleus sampling
    }
)

print(response['response'])

