import streamlit as st
# from transformers import pipeline
# import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from bs4 import BeautifulSoup
import requests
from hugchat import hugchat
from hugchat.login import Login
import mediapy as mp
from better_profanity import Profanity
from diffusers import EulerAncestralDiscreteScheduler as EAD
from diffusers import StableDiffusionPipeline as sdp
# video
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from IPython.display import HTML
from base64 import b64encode
import datetime
import os
import subprocess
from docquery import document, pipeline


# qa_model = pipeline("question-answering")

def has_profanity(text):
    return Profanity().contains_profanity(text)


def filter_text(text):
    while has_profanity(text):
        text = input("Please provide an alternative prompt: ")
    return text


def docquery1(uploaded_file):
    p = pipeline('document-question-answering')
    st.write(os.getcwd(), uploaded_file)
    doc = document.load_document(uploaded_file)
    for q in ["what are the Components of Cloud Computing Architecture?", "What is cloud reference model?"]:
        st.write(q, p(question=q, **doc.context))
    # st.write(res)


def hfgptinput(style, topic):
    st.write("//logging into hugchat ")
    email = 'ahmedmuzammil.ai@gmail.com'
    passwd = 'Teamsmc12#'
    sign = Login(email, passwd)
    cookies = sign.login()
    st.write("//logged into hugchat ")
    # Save cookies to the local directory
    cookie_path_dir = "./cookies_snapshot"
    sign.saveCookiesToDir(cookie_path_dir)

    # Load cookies when you restart your program:
    # sign = login(email, None)
    # cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.

    # Create a ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

    # st.write("querying")

    # non stream response
    prompt = style + topic
    # query_result = chatbot.query(prompt)

    # query_result = chatbot.query("tell me something about cop28")
    # print(query_result) # or query_result.text or query_result["text"]
    # st.write(query_result.text)

    # # stream response
    # for resp in chatbot.query(
    #     "Hello",
    #     stream=True
    # ):
    #     print(resp)
    st.write("//Using web search to fetch real-time data")
    st.write("//generating hook..")

    # Use web search *new
    query_result = chatbot.query(prompt, web_search=True)
    print(query_result)  # or query_result.text or query_result["text"]
    for source in query_result.web_search_sources:
        print(source.link)
        print(source.title)
        print(source.hostname)

    # Create a new conversation
    id = chatbot.new_conversation()
    chatbot.change_conversation(id)

    # Get conversation list
    conversation_list = chatbot.get_conversation_list()

    # Switch model (default: meta-llama/Llama-2-70b-chat-hf. )
    chatbot.switch_llm(0)  # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
    chatbot.switch_llm(1)  # llama 2.0
    return query_result


def video(prompt):
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16,
                                             variant="fp16")
    st.write("// image gen model begin.. model = damo-vilab/text-to-video-ms-1.7b")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    os.system('mkdir /videos')
    # prompt = 'cutting a cake' #@param {type:"string"}
    prompt = prompt
    st.write("// generting video for:", prompt)
    # st.write(os.getcwd())
    negative_prompt = "low quality"  # @param {type:"string"}
    num_frames = 30  # @param {type:"raw"}
    video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, num_frames=num_frames).frames
    output_video_path = export_to_video(video_frames)

    new_video_path = f'/videos/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
    os.system('ffmpeg -y -i {output_video_path} -c:v libx264 -c:a aac -strict -2 {new_video_path} >/dev/null 2>&1')

    # output_video_path = "input_video.mp4"  # Replace with your input video path
    # new_video_path = "output_video.mp4"    # Replace with your desired output video path

    # Construct the ffmpeg command as a list of arguments
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", output_video_path,  # Input video file
        "-c:v", "libx264",  # Video codec
        "-c:a", "aac",  # Audio codec
        "-strict", "-2",  # Allow non-standard AAC codec
        new_video_path  # Output video file
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Video conversion successful. Output saved to: {new_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
    # st.write(output_video_path, '->', new_video_path)
    st.write("Video generated:")
    video_file = open(new_video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def images(prompt):
    model = "dreamlike-art/dreamlike-photoreal-2.0"
    st.write("// image gen model begin.. model = ", model)

    scheduler = EAD.from_pretrained(model, subfolder="scheduler")

    pipe = sdp.from_pretrained(
        model,
        scheduler=scheduler
    )
    device = "cuda"

    pipe = pipe.to(device)
    prompt = prompt
    st.write("// generting images for:", prompt)
    num_images = 3
    filtered_input = filter_text(prompt)
    images = pipe(
        filtered_input,
        height=512,
        width=512,
        num_inference_steps=30,  # more no of steps,  better results
        guidance_scale=9,  # more no of steps,  better results
        num_images_per_prompt=num_images

    ).images
    st.image(images)


def question_from_pdf():
    st.subheader("Question from Document/Notes (PDF)")
    uploaded_file = st.file_uploader("Upload your notes...", type=["pdf", "txt", "docx"])
    # Add your code for image question extraction here
    if uploaded_file is not None:
        # Display the image
        st.write(uploaded_file)
        # docquery1(str(uploaded_file.name))
        # st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("Please upload an image.")
    # Add your code for PDF question extraction here


# Page for questions from an image
def question_from_image():
    st.subheader("Question from Image Notes")
    # st.title("ask answers from images:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    # Add your code for image question extraction here
    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("Please upload an image.")


# Page for concept explanation using hooks
def concept_explanation():
    st.subheader("Concept Explanation Using Hooks")
    topic = st.text_input("Enter the topic:")

    with st.sidebar:
        st.markdown("## Style")
        style_selected = st.radio("", ["Story", "Question", "Image", "Video", "Real Event", "Surprising Fact"])

        st.markdown("## Grade Level")
        grade_level_selected = st.radio("", ["Primary", "Secondary"])

        st.markdown("## Tone")
        tone_selected = st.radio("", ["Humorous", "Serious"])
    if style_selected == "Story":
        if st.button("Generate Story Hook") and topic:
            append = 'generate a story hook for the topic '
            generated_hook_hf = hfgptinput(append, topic)
            # generated_hook_gpt2 = generate_hook(topic, style=style_selected, grade_level=grade_level_selected, tone=tone_selected)
            st.subheader("Generated Hook:")
            st.write(generated_hook_hf.text)
    if style_selected == "Surprising Fact":
        if st.button("Generate a Hook with surprising fact") and topic:
            append = 'generate a hook with surprising fact for the topic'
            generated_hook_hf = hfgptinput(append, topic)
            # generated_hook_gpt2 = generate_hook(topic, style=style_selected, grade_level=grade_level_selected, tone=tone_selected)
            st.subheader("Generated Hook:")
            st.write(generated_hook_hf.text)
    if style_selected == "Real Event":
        if st.button("Generate a Hook with a real-world event") and topic:
            append = 'generate a hook referring to a current real world event for the topic'
            generated_hook_hf = hfgptinput(append, topic)
            # generated_hook_gpt2 = generate_hook(topic, style=style_selected, grade_level=grade_level_selected, tone=tone_selected)
            st.subheader("Generated Hook:")
            st.write(generated_hook_hf.text)
    if style_selected == "Image":
        if st.button("Generate images") and topic:
            prompt = topic
            images(prompt)
    if style_selected == "Video":
        if st.button("Generate video") and topic:
            prompt = topic
            video(prompt)
    if style_selected == "Question":
        if st.button("Generate Question Hook") and topic:
            append = 'generate a question hook for the topic '
            generated_hook_hf = hfgptinput(append, topic)
            # generated_hook_gpt2 = generate_hook(topic, style=style_selected, grade_level=grade_level_selected, tone=tone_selected)
            st.subheader("Generated Hook:")
            st.write(generated_hook_hf.text)


def career_guide():
    topic = st.text_input("Ask away:")
    if st.button("Ask") and topic:
        append = "career guidance on: "
        generated_hook_hf = hfgptinput(append, topic)
        st.write("generated.")
        st.write(generated_hook_hf.text)


def study_mode():
    st.title("Preferred mode of assistance")
    selected_page = st.radio("Select one",
                             ["Question from PDF", "Question from Image", "Concept Explanation", "Career Guidance"])

    if selected_page == "Question from PDF":
        question_from_pdf()
    elif selected_page == "Question from Image":
        question_from_image()
    elif selected_page == "Concept Explanation":
        concept_explanation()
    elif selected_page == "Career Guidance":
        career_guide()

    # st.subheader("Your answer will appear here: ")
    # output_box = st.empty()
    # output_box.write("answer..")
    if st.button("Speak the output out loud!"):
        st.write("using Eleven Labs API to generate AI Voice..")
    language_options = ["Assamese", "Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam",
                        "Marathi", "Oriya", "Punjabi", "Tamil", "Telugu"]

    # Create a radio button for language selection

    selected_language = st.selectbox("Choose a language:", language_options)

    if st.button("Translate"):
        st.write(f"Your translated response in {selected_language} will appear here:")

    # user_question = st.text_input("Ask your question:")

    # In this section, you would process the user_question and provide an answer
    # For now, let's just display a placeholder answer

    # Check if an image has been uploaded

    # st.title("Answer")
    # st.write("Answer to your question will appear here.")

    # edhook app from here

    # Collect user inputs

    # Main section
    # if st.button("Generate Hook") and topic:
    #     generated_hook_gpt2 = generate_hook(topic, style=style_selected, grade_level=grade_level_selected, tone=tone_selected)
    #     st.subheader("Generated Hugging Face Hook:")
    #     st.write(generated_hook_gpt2)

    # if st.button("Web Scrape Example") and topic:
    # scraped_text = web_scrape_example(topic)
    # st.write("Web Scraped Text:")
    # st.write(scraped_text)

    # Allow the user to ask queries on the scraped data
    # question = st.text_input("Ask a question about the scraped data:")
    # if question:
    #     answer = qa_model(question=question, context=scraped_text)
    #     st.write("Answer:")
    #     st.write(answer['answer'])


def wellness_mode():
    st.title("I'm here to listen and help. What's on your mind?")
    user_input = st.text_area("Type your thoughts here:")

    # In this section, you would pass user_input to LLM or any other model for processing
    # For now, let's just display a placeholder response
    st.title("Response")
    st.write("Response from LLM or other model will appear here.")


# Add a sidebar to the Streamlit app
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose a mode:", ["Study", "Wellness"])

# Display the appropriate mode based on the user's selection
if mode == "Study":

    study_mode()
elif mode == "Wellness":
    wellness_mode()


# qa_model = pipeline("question-answering")


def has_profanity(text):
    return Profanity().contains_profanity(text)


def filter_text(text):
    while has_profanity(text):
        text = input("Please provide an alternative prompt: ")
    return text


def generate_hook(prompt, model="gpt2", style="story", grade_level="primary", tone="humorous"):
    # Customize the prompt based on selected style, grade level, and tone
    prompt = f"Create a {style.lower()} hook for {grade_level} students with a {tone.lower()} tone about {prompt}"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=200, truncation=True)
    output = model.generate(input_ids, max_length=150, temperature=0.7, num_beams=5, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def images():
    model = "dreamlike-art/dreamlike-photoreal-2.0"
    scheduler = EAD.from_pretrained(model, subfolder="scheduler")

    pipe = sdp.from_pretrained(
        model,
        scheduler=scheduler
    )
    device = "cuda"

    pipe = pipe.to(device)
    prompt = "baking a cake"
    st.write("generting imgs for:", prompt)
    num_images = 3
    filtered_input = filter_text(prompt)
    images = pipe(
        filtered_input,
        height=512,
        width=512,
        num_inference_steps=30,  # more no of steps,  better results
        guidance_scale=9,  # more no of steps,  better results
        num_images_per_prompt=num_images

    ).images
    st.image(images)


def video():
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16,
                                             variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    os.system('mkdir /videos')
    prompt = 'cutting a cake'  # @param {type:"string"}
    st.write("generting video for:", prompt)
    st.write(os.getcwd())
    negative_prompt = "low quality"  # @param {type:"string"}
    num_frames = 30  # @param {type:"raw"}
    video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, num_frames=num_frames).frames
    output_video_path = export_to_video(video_frames)

    new_video_path = f'/videos/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
    os.system('ffmpeg -y -i {output_video_path} -c:v libx264 -c:a aac -strict -2 {new_video_path} >/dev/null 2>&1')

    st.write(output_video_path, '->', new_video_path)
    video_file = open(output_video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def web_scrape_example(topic):
    url = f'https://en.wikipedia.org/wiki/{topic}'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            paragraphs = content_div.find_all('p')  # Extracting paragraphs
            content = '\n'.join([p.get_text() for p in paragraphs])
            return content
        else:
            return "No content found on the Wikipedia page."
    else:
        return "Failed to fetch content. Check your internet connection or try a different topic."
