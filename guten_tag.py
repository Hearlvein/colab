# %% [markdown]
# # Fine-tuning GPT-2 on Sci-Fi & Poetry with LoRA
# Adapting from: https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face

# %%
# Install dependencies
%pip install -q datasets transformers accelerate peft bitsandbytes einops
%pip install -q beautifulsoup4 requests gutenbergpy

# %%
import os
import json
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from gutenbergpy.textget import get_text_by_id
from gutenbergpy.gutenbergcache import GutenbergCache
from tqdm import tqdm

# %%
# Utility: Extract book IDs from a Gutenberg bookshelf
def get_book_ids_from_bookshelf(url, limit=10):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    book_links = soup.select("li.booklink a.link")
    book_ids = []
    for link in book_links:
        href = link.get("href")
        if href.startswith("/ebooks/"):
            book_id = href.split("/")[-1]
            if book_id.isdigit():
                book_ids.append(int(book_id))
                if len(book_ids) == limit:
                    break
    return book_ids

# Download and cache books
def download_books(book_ids, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print("Loading Gutenberg metadata cache...")
    cache = GutenbergCache.get_cache()
    for book_id in book_ids:
        output_path = os.path.join(output_folder, f"{book_id}.txt")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Book {book_id} already exists. Skipping.")
            continue
        try:
            print(f"Downloading book ID {book_id}...")
            text_bytes = get_text_by_id(book_id)
            text_str = text_bytes.decode("utf-8", errors="ignore")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_str)
        except Exception as e:
            print(f"Failed to download {book_id}: {e}")

def download_books_to_dataset(bookshelf_url, genre, limit=10, base_folder="gutenberg_dataset"):
    folder = os.path.join(base_folder, genre)
    ids = get_book_ids_from_bookshelf(bookshelf_url, limit)
    download_books(ids, folder)

# %%
# Define bookshelves
bookshelves = {
    "fiction": "https://www.gutenberg.org/ebooks/bookshelf/480",
    "poetry": "https://www.gutenberg.org/ebooks/bookshelf/60",
}

# Download books by genre
for genre, url in bookshelves.items():
    download_books_to_dataset(url, genre, limit=10)

# %%
# Clean and structure the dataset
HEADER_PATTERN = re.compile(r"\*{3}\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*{3}", re.IGNORECASE | re.DOTALL)
FOOTER_PATTERN = re.compile(r"\*{3}\s*END OF THIS PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

def clean_text(text):
    text = HEADER_PATTERN.sub("", text)
    text = FOOTER_PATTERN.sub("", text)
    return text.strip()

def build_jsonl_dataset(input_dirs, output_file):
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"{output_file} exists. Skipping creation.")
        return
    with open(output_file, "w", encoding="utf-8") as out_f:
        for genre, folder in input_dirs.items():
            for path in Path(folder).rglob("*.txt"):
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                    cleaned = clean_text(raw)
                    if cleaned:
                        json.dump({"source": genre, "filename": path.name, "text": cleaned}, out_f, ensure_ascii=False)
                        out_f.write("\n")
                except Exception as e:
                    print(f"Error processing {path}: {e}")

# Prepare dataset
INPUT_DIRS = {
    "fiction": "gutenberg_dataset/fiction",
    "poetry": "gutenberg_dataset/poetry",
}
OUTPUT_FILE = "gutenberg_dataset.jsonl"
build_jsonl_dataset(INPUT_DIRS, OUTPUT_FILE)

# %%
# Load dataset
from datasets import Dataset
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]
dataset = Dataset.from_list(data)
print(f"Loaded {len(dataset)} samples.")

# %%
# Filter and tokenize
from transformers import AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    result = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "filename", "source"])

# %%
# LoRA setup
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

base_model = AutoModelForCausalLM.from_pretrained(model_name)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# %%
# Prepare training
output_dir = "./gpt2-lora-sci-fi-poetry"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# %%
# Train model
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
# Generate text
from transformers import pipeline

generator = pipeline("text-generation", model=output_dir, tokenizer=output_dir)

prompt = "Beneath the rusted moons of Elarion, the last poet of Earth recited verses to the wind."

output = generator(
    prompt,
    max_new_tokens=300,
    temperature=0.95,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
)

print("\nGenerated Poetic Sci-Fi Story:\n")
print(output[0]["generated_text"])
