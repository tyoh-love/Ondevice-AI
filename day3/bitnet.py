import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
print("Loading BitNet model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" if torch.cuda.is_available() else None
)

# Move model to appropriate device
if device.type == "cuda":
    model = model.to(device)

print("Model loaded successfully!")

# Initialize conversation history
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
]

print("\nBitNet Chat Interface")
print("=====================")
print("Type 'exit' or 'quit' to end the conversation")
print("Type 'clear' to clear the conversation history")
print("Type 'help' for more commands\n")

# Interactive chat loop
while True:
    # Get user input
    user_input = input("You: ").strip()
    
    # Check for special commands
    if user_input.lower() in ['exit', 'quit']:
        print("\nGoodbye!")
        break
    elif user_input.lower() == 'clear':
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        print("\nConversation history cleared.")
        continue
    elif user_input.lower() == 'help':
        print("\nAvailable commands:")
        print("  exit/quit - End the conversation")
        print("  clear     - Clear conversation history")
        print("  help      - Show this help message\n")
        continue
    elif not user_input:
        continue
    
    # Add user message to conversation
    messages.append({"role": "user", "content": user_input})
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print("BitNet: ", end="", flush=True)
    chat_outputs = model.generate(**chat_input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
    print(response)
    
    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": response})
