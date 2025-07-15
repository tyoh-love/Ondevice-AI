#!/usr/bin/env python
# encoding: utf-8

# Simplified MiniCPM-O Ollama Demo - Cleaned up version
# Original: https://github.com/OpenBMB/MiniCPM-o/blob/main/web_demos/minicpm-o_2.6/chatbot_web_demo_o2.6.py

import argparse
import gradio as gr
from PIL import Image
from decord import VideoReader
import io
import os
import copy
import base64
import json
import traceback
import re
import modelscope_studio as mgr
import ollama

# README: How to run demo with Ollama
# Ensure Ollama is running with the model: ollama run openbmb/minicpm-o2.6:latest

# Arguments
parser = argparse.ArgumentParser(description='MiniCPM-O Ollama Demo')
parser.add_argument('--model', type=str, default="openbmb/minicpm-o2.6:latest", help="Ollama model name")
parser.add_argument('--ollama-host', type=str, default="http://localhost:11434", help="Ollama API host")
args = parser.parse_args()

model_name = 'MiniCPM-o 2.6 (Ollama)'
ollama_model = args.model

# Test Ollama connection
try:
    ollama.list()
    print(f"Successfully connected to Ollama. Using model: {ollama_model}")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("Please ensure Ollama is running with: ollama serve")
    exit(1)

# Constants
ERROR_MSG = "Error, please retry"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS

def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS

def check_mm_type(mm_file):
    """Check if file is image or video"""
    if hasattr(mm_file, 'path'):
        path = mm_file.path
    else:
        path = mm_file.file.path
    if is_image(path):
        return "image"
    if is_video(path):
        return "video"
    return None

def process_image(image):
    """Simple image processing for Ollama"""
    if not isinstance(image, Image.Image):
        if hasattr(image, 'path'):
            image = Image.open(image.path).convert("RGB")
        else:
            image = Image.open(image.file.path).convert("RGB")
    return image

def process_video(video):
    """Simple video processing - extract a few key frames"""
    if hasattr(video, 'path'):
        vr = VideoReader(video.path)
    else:
        vr = VideoReader(video.file.path)
    
    # Extract only 4 key frames for efficiency
    total_frames = len(vr)
    max_frames = min(4, total_frames)
    
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // max_frames
        frame_indices = list(range(0, total_frames, step))[:max_frames]
    
    frames = []
    for idx in frame_indices:
        frame_array = vr[idx].asnumpy()
        frame_image = Image.fromarray(frame_array)
        frames.append(process_image(frame_image))
    
    return frames

def simple_chat(messages, use_sampling=True):
    """Simplified chat function for Ollama"""
    try:
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', [])
            
            if isinstance(content, list):
                text_parts = []
                images = []
                
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, Image.Image):
                        buffered = io.BytesIO()
                        item.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_base64)
                
                message_text = ' '.join(text_parts)
                
                if images:
                    ollama_messages.append({
                        'role': role,
                        'content': message_text,
                        'images': images
                    })
                else:
                    ollama_messages.append({
                        'role': role,
                        'content': message_text
                    })
            else:
                ollama_messages.append({
                    'role': role,
                    'content': str(content)
                })
        
        # Simple parameter setup
        options = {
            'temperature': 0.7 if use_sampling else 0.1,
            'top_p': 0.8,
            'top_k': 100,
            'num_predict': 2048,
            'repeat_penalty': 1.05,
            'num_ctx': 8192
        }
        
        response = ollama.chat(
            model=ollama_model,
            messages=ollama_messages,
            options=options
        )
        
        answer = response['message']['content']
        return answer
    except Exception as e:
        print(f"Error in chat: {e}")
        traceback.print_exc()
        return ERROR_MSG

def process_files(files):
    """Process uploaded files (images/videos)"""
    processed_files = []
    for file in files:
        if check_mm_type(file) == 'image':
            processed_files.append(process_image(file))
        elif check_mm_type(file) == 'video':
            processed_files.extend(process_video(file))
    return processed_files

def create_message(question_text, files):
    """Create a simple message with text and files"""
    # Clean up text
    cleaned_text = re.sub(r"\[mm_media\]\d+\[/mm_media\]", "", question_text).strip()
    
    message = []
    if cleaned_text:
        message.append(cleaned_text)
    
    # Add processed files
    processed_files = process_files(files)
    message.extend(processed_files)
    
    return message

def count_file_types(files):
    """Count images and videos in uploaded files"""
    images_cnt = sum(1 for file in files if check_mm_type(file) == "image")
    videos_cnt = sum(1 for file in files if check_mm_type(file) == "video")
    return images_cnt, videos_cnt

def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    try:
        # Try newer API with button props
        return mgr.MultimodalInput(
            upload_image_button_props={'label': 'Upload Image', 'disabled': upload_image_disabled, 'file_count': 'multiple'},
            upload_video_button_props={'label': 'Upload Video', 'disabled': upload_video_disabled, 'file_count': 'single'},
            submit_button_props={'label': 'Submit'}
        )
    except TypeError:
        # Fall back to simpler API for older versions
        print("Note: Using legacy MultimodalInput API. Some features may be limited.")
        return mgr.MultimodalInput()

def respond(_question, _chat_bot, _app_cfg, use_sampling=True):
    """Simplified response function"""
    context = _app_cfg['ctx'].copy()
    context.append({'role': 'user', 'content': create_message(_question.text, _question.files)})

    images_cnt, videos_cnt = count_file_types(_question.files)
    total_images = _app_cfg['images_cnt'] + images_cnt
    total_videos = _app_cfg['videos_cnt'] + videos_cnt
    
    # Check file limits
    if total_videos > 1 or (total_videos == 1 and total_images > 0):
        gr.Warning("Only supports single video file input right now!")
        return _question, _chat_bot, _app_cfg

    # Get response from Ollama
    answer = simple_chat(context, use_sampling)
    
    if answer != ERROR_MSG:
        context.append({"role": "assistant", "content": [answer]})
        _chat_bot.append((_question, answer))
        _app_cfg['ctx'] = context
        _app_cfg['images_cnt'] = total_images
        _app_cfg['videos_cnt'] = total_videos

    upload_image_disabled = total_videos > 0
    upload_video_disabled = total_videos > 0 or total_images > 0
    return create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg

def fewshot_add_demonstration(_image, _user_message, _assistant_message, _chat_bot, _app_cfg):
    """Add demonstration example for few-shot learning"""
    ctx = _app_cfg["ctx"]
    message_item = []
    
    if _image is not None:
        image = Image.open(_image).convert("RGB")
        ctx.append({"role": "user", "content": [process_image(image), _user_message]})
        message_item.append({"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]})
    else:
        if _user_message:
            ctx.append({"role": "user", "content": [_user_message]})
            message_item.append({"text": _user_message, "files": []})
        else:
            message_item.append(None)
            
    if _assistant_message:
        ctx.append({"role": "assistant", "content": [_assistant_message]})
        message_item.append({"text": _assistant_message, "files": []})
    else:
        message_item.append(None)

    _chat_bot.append(message_item)
    return None, "", "", _chat_bot, _app_cfg

def fewshot_respond(_image, _user_message, _chat_bot, _app_cfg, use_sampling=True):
    """Simplified few-shot response function"""
    user_message_contents = []
    context = _app_cfg["ctx"].copy()
    
    if _image:
        image = Image.open(_image).convert("RGB")
        user_message_contents.append(process_image(image))
    if _user_message:
        user_message_contents.append(_user_message)
    if user_message_contents:
        context.append({"role": "user", "content": user_message_contents})

    answer = simple_chat(context, use_sampling)
    
    if answer != ERROR_MSG:
        context.append({"role": "assistant", "content": [answer]})
        _app_cfg['ctx'] = context

    if _image:
        _chat_bot.append([
            {"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]},
            {"text": answer, "files": []}        
        ])
    else:
        _chat_bot.append([
            {"text": _user_message, "files": [_image]},
            {"text": answer, "files": []}        
        ])
    
    return None, '', '', _chat_bot, _app_cfg

def regenerate_button_clicked(_question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg, use_sampling=True):
    """Simplified regenerate function"""
    if len(_chat_bot) <= 1 or not _chat_bot[-1][1]:
        gr.Warning('No question for regeneration.')
        return '', _image, _user_message, _assistant_message, _chat_bot, _app_cfg
        
    if _app_cfg["chat_type"] == "Chat":
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        
        # Update counts
        images_cnt, videos_cnt = count_file_types(_question.files)
        _app_cfg['images_cnt'] -= images_cnt
        _app_cfg['videos_cnt'] -= videos_cnt
        
        _question, _chat_bot, _app_cfg = respond(_question, _chat_bot, _app_cfg, use_sampling)
        return _question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg
    else: 
        last_message = _chat_bot[-1][0]
        last_image = None
        last_user_message = ''
        if last_message.text:
            last_user_message = last_message.text
        if last_message.files:
            last_image = last_message.files[0].file.path
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        _image, _user_message, _assistant_message, _chat_bot, _app_cfg = fewshot_respond(last_image, last_user_message, _chat_bot, _app_cfg, use_sampling)
        return _question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg

def flushed():
    return gr.update(interactive=True)

def clear(txt_message, chat_bot, app_session):
    """Clear chat history and reset session"""
    txt_message.files.clear()
    txt_message.text = ''
    chat_bot = copy.deepcopy(init_conversation)
    app_session['ctx'] = []
    app_session['images_cnt'] = 0
    app_session['videos_cnt'] = 0
    return create_multimodal_input(), chat_bot, app_session, None, '', ''

def select_chat_type(_tab, _app_cfg):
    _app_cfg["chat_type"] = _tab
    return _app_cfg

init_conversation = [
    [
        None,
        {
            "text": "You can talk to me now",
            "flushing": False
        }
    ],
]

css = """
video { height: auto !important; }
.example label { font-size: 16px;}
"""

introduction = """
## Features:
1. Chat with single image
2. Chat with multiple images
3. Chat with video
4. In-context few-shot learning

Click `How to use` tab to see examples.
"""

with gr.Blocks(css=css) as demo:
    with gr.Tab(model_name):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown(value=introduction)
                use_sampling = gr.Checkbox(value=True, label="Use Sampling (vs Deterministic)")
                regenerate = gr.Button("Regenerate")
                clear_button = gr.Button("Clear History")

            with gr.Column(scale=3, min_width=500):
                app_session = gr.State({'ctx':[], 'images_cnt': 0, 'videos_cnt': 0, 'chat_type': 'Chat'})
                chat_bot = mgr.Chatbot(label=f"Chat with {model_name}", value=copy.deepcopy(init_conversation), height=600, flushing=False, bubble_full_width=False)
                
                with gr.Tab("Chat") as chat_tab:
                    txt_message = create_multimodal_input()
                    chat_tab_label = gr.Textbox(value="Chat", interactive=False, visible=False)

                    txt_message.submit(
                        respond,
                        [txt_message, chat_bot, app_session, use_sampling], 
                        [txt_message, chat_bot, app_session]
                    )

                with gr.Tab("Few Shot") as fewshot_tab:
                    fewshot_tab_label = gr.Textbox(value="Few Shot", interactive=False, visible=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="filepath", sources=["upload"])
                        with gr.Column(scale=3):
                            user_message = gr.Textbox(label="User")
                            assistant_message = gr.Textbox(label="Assistant")
                            with gr.Row():
                                add_demonstration_button = gr.Button("Add Example")
                                generate_button = gr.Button(value="Generate", variant="primary")
                    add_demonstration_button.click(
                        fewshot_add_demonstration,
                        [image_input, user_message, assistant_message, chat_bot, app_session],
                        [image_input, user_message, assistant_message, chat_bot, app_session]
                    )
                    generate_button.click(
                        fewshot_respond,
                        [image_input, user_message, chat_bot, app_session, use_sampling],
                        [image_input, user_message, assistant_message, chat_bot, app_session]
                    )

                chat_tab.select(
                    select_chat_type,
                    [chat_tab_label, app_session],
                    [app_session]
                )
                chat_tab.select( # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )
                fewshot_tab.select(
                    select_chat_type,
                    [fewshot_tab_label, app_session],
                    [app_session]
                )
                fewshot_tab.select( # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )
                chat_bot.flushed(
                    flushed,
                    outputs=[txt_message]
                )
                regenerate.click(
                    regenerate_button_clicked,
                    [txt_message, image_input, user_message, assistant_message, chat_bot, app_session, use_sampling],
                    [txt_message, image_input, user_message, assistant_message, chat_bot, app_session]
                )
                clear_button.click(
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )

    with gr.Tab("How to use"):
        with gr.Column():
            with gr.Row():
                image_example = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/m_bear2.gif", label='1. Chat with single or multiple images', interactive=False, width=400, elem_classes="example")
                example2 = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/video2.gif", label='2. Chat with video', interactive=False, width=400, elem_classes="example")
                example3 = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/fshot.gif", label='3. Few shot', interactive=False, width=400, elem_classes="example")

# Launch
if __name__ == "__main__":
    print(f"Starting MiniCPM-O Ollama Demo with model: {ollama_model}")
    print("Make sure Ollama is running and the model is pulled:")
    print(f"  ollama pull {ollama_model}")
    print(f"  ollama run {ollama_model}")
    demo.launch(share=False, debug=True, show_api=False, server_port=8000, server_name="0.0.0.0")