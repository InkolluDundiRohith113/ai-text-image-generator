from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import gradio as gr


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key = HF_TOKEN
)

def generate_text(prompt):
    response = client.chat_completion(
        model= "Qwen/Qwen3-4B-Instruct-2507" ,
        messages=[
            {
                "role" : "user" ,
                "content" : prompt
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content


def generate_image(prompt):
    image = client.text_to_image(
        prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )
    return image





def chat(message, history):
    text = generate_text(message)

    image_words = [
        "generate image",
        "create image",
        "make an image",
        "generate a picture",
        "create a picture",
        "draw",
        "image of",
        "picture of",
        "generate a pic"
    ]
    wants_image = any(word in message.lower() for word in image_words)

    if wants_image:
        image = generate_image(message)


        return [
        {
            "role" : "assistant",
            "content" : text
        },
        {
            "role" : "assistant",
            "content" : gr.Image(value = image)
        }

    ]
    return text


demo = gr.ChatInterface(
    fn=chat,
    title="AI ASSIANT"
)

demo.launch()

