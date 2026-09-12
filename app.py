from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN
)


def generate_text(prompt):
    response = client.chat_completion(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[
            {
                "role": "user",
                "content": prompt
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


st.title("AI ASSISTANT")

message = st.chat_input("Ask something...")


if message:

    with st.chat_message("user"):
        st.write(message)

    text_words = [
        "what",
        "who",
        "why",
        "when",
        "where",
        "explain",
        "tell me",
        "about"
    ]

    image_words = [
        "image",
        "picture",
        "photo",
        "draw",
        "generate",
        "create",
        "make",
        "show"
    ]

    both_words = [
        "also",
        "along with",
        "with explanation",
        "and explain",
        "and",
        ", generate "
    ]

    msg = message.lower()

    if (
        any(word in msg for word in image_words)
        and
        any(word in msg for word in both_words)
    ):

        with st.chat_message("assistant"):

            text = generate_text(
                f"""
                Answer only the information part of the user's request.
                Do not talk about images or image generation.

                User request:
                {message}
                """
            )

            st.write(text)

            with st.spinner("Generating image..."):
                image = generate_image(message)

            st.image(
                image,
                caption="Generated Image",
                use_container_width=True
            )

    elif any(word in msg for word in image_words):

        with st.chat_message("assistant"):

            with st.spinner("Generating image..."):
                image = generate_image(message)

            st.image(
                image,
                caption="Generated Image",
                use_container_width=True
            )

    elif any(word in msg for word in text_words):

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                text = generate_text(message)

            st.write(text)

    else:

        with st.chat_message("assistant"):
            st.write("Wrong input. I cannot do that.")