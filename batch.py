import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates
from tqdm import tqdm
import time
import re

torch.set_float32_matmul_precision("high")

# Configuration options
OVERWRITE = True  # Boolean option to allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
BATCH_SIZE = 7  # Adjust based on VRAM. For 24GB VRAM, 7 is good. For 6GB, set to 1.
PRINT_PROCESSING_STATUS = False  # Option to print processing status of images
PRINT_CAPTIONS = False  # Option to print captions to the console
OUTPUT_KEYWORDS = False  # Option to control whether to output comma-delimited keywords (True) or a normal description (False, default)

PROMPT_NORMAL = 'Describe this image.'
PROMPT_KEYWORDS = 'List up to 10 of the most important, distinct, and descriptive elements in this image as keywords. Focus on key objects, scenery, colors, and relevant visual features. Do not include any numbers or repeated words.'

#MODEL_PATH = 'NVEagle/Eagle-X5-13B-Chat'
MODEL_PATH = 'NVEagle/Eagle-X5-7B'
CONV_MODE = "vicuna_v1"

# Load the model
def download_and_load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device available: {device}')

    model_name = get_model_name_from_path(MODEL_PATH)
    
    print(f"Loading model {model_name}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None, model_name, False, False)
    
    model.to(device)
    print("Model loaded.")
    return model, tokenizer, image_processor

# Load image paths recursively
def load_image_paths_recursive(folder_path: str):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return (
        path for path in Path(folder_path).rglob("*")
        if path.suffix.lower() in valid_extensions and (OVERWRITE or not path.with_suffix('.txt').exists())
    )

# Clean the captions by removing numbers, leading punctuation, hyphens, extra commas, duplicates, and unnecessary spaces
def clean_caption(caption: str) -> str:
    # Remove square brackets if they exist
    caption = caption.replace("[", "").replace("]", "")
    
    # Remove numbers followed by periods (e.g., "1. Dragon")
    caption = re.sub(r'\d+\.\s*', '', caption)
    
    # Split by commas and newlines, strip spaces, and remove empty strings
    keywords = [word.strip() for word in re.split(r'[,\n]', caption) if word.strip()]
    
    # Remove leading punctuation (e.g., ". Dragon", "- Castle") from each keyword
    keywords = [re.sub(r'^[\.\-\s]+', '', word) for word in keywords]
    
    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))
    
    # Return cleaned, comma-separated string
    return ", ".join(keywords)

# Process a batch of images and return captions
def run_model_batch(image_paths, model, tokenizer, image_processor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use the appropriate prompt based on OUTPUT_KEYWORDS
    input_prompt = PROMPT_KEYWORDS if OUTPUT_KEYWORDS else PROMPT_NORMAL
    
    if model.config.mm_use_im_start_end:
        input_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_prompt
    else:
        input_prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_prompt
    
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    inputs = []
    for image_path in image_paths:
        if PRINT_PROCESSING_STATUS:
            print(f"Processing image: {image_path}")
        with Image.open(image_path).convert("RGB") as img:
            inputs.append(process_images([img], image_processor, model.config)[0])

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.to(device)
    image_tensors = torch.cat([img.unsqueeze(0) for img in inputs]).to(dtype=torch.float16, device=device, non_blocking=True)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            images=image_tensors,
            image_sizes=[img.size for img in inputs],
            do_sample=True,
            temperature=0.2,
            top_p=0.5,
            num_beams=1,
            max_new_tokens=512,
            use_cache=False
        )

    results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    #print(f"{results}")
    
    if OUTPUT_KEYWORDS:
        # Clean up each caption to remove square brackets, double commas, and duplicates
        return [clean_caption(result.strip()) for result in results]
    else:
        # Return normal descriptions without cleaning for keywords
        return [result.strip() for result in results]

# Process images in batches
def process_images_recursive(paths, model, tokenizer, image_processor, batch_size=BATCH_SIZE):
    start_time = time.time()
    total_images = 0

    path_list = list(paths)
    num_batches = len(path_list) // batch_size + (1 if len(path_list) % batch_size > 0 else 0)

    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch = path_list[i*batch_size:(i+1)*batch_size]
        captions = run_model_batch(batch, model, tokenizer, image_processor)
        for path, caption in zip(batch, captions):
            # Ensure we clean and format caption here before writing to the file
            caption = f"{PREPEND_STRING}{caption}{APPEND_STRING}"
            if PRINT_CAPTIONS:
                print(f"Caption for {path}: {caption}")
            path.with_suffix('.txt').write_text(caption)
            total_images += 1

    total_time = time.time() - start_time
    return total_images, total_time

# Main execution
model, tokenizer, image_processor = download_and_load_model()

# Process images in the /input/ folder
folder_path = Path(__file__).parent / "input"
total_images, total_time = process_images_recursive(load_image_paths_recursive(folder_path), model, tokenizer, image_processor, batch_size=BATCH_SIZE)

print(f"Total images captioned: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")

# Fix for divide-by-zero when calculating average time per image
if total_images > 0:
    print(f"Average time per image: {total_time / total_images:.2f} seconds")
else:
    print("No images were processed, so no average time to display.")
