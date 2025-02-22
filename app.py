import openai
import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import tempfile
import wave
import time
import re
import os

# ✅ Load API key from secrets or environment variables
api_key = st.secrets.get("openai", {}).get("api_key", os.getenv("OPENAI_API_KEY"))

# ✅ Debugging: Check if API key is loaded
st.write(f"🔍 **API Key Loaded?** {'✅ Yes' if api_key else '❌ No'}")

# 🚨 Stop execution if API key is missing
if not api_key:
    st.error("🚨 OpenAI API key is missing! Check `.streamlit/secrets.toml` or environment variables.")
    st.stop()

# ✅ Assign the API key correctly
openai.api_key = api_key  # ✅ Correct way to assign API key

# ✅ Load Korean Profanity List (if missing, use default empty list)
korean_profanity_list = st.secrets.get("korean_profanity", {}).get("words", os.getenv("KOREAN_PROFANITY", "")).split(", ")

# ✅ Load AI Prompt Text
prompt_text = st.secrets.get("ai", {}).get("prompt_text", os.getenv("AI_PROMPT_TEXT", ""))

# Audio queue for real-time recording
q = queue.Queue()

# 🔹 Initialize session state variables **at the start**
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "response_count" not in st.session_state:
    st.session_state.response_count = 0
if "strike_count" not in st.session_state:
    st.session_state.strike_count = 0  # Track user warnings
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "custom_prompts" not in st.session_state:
    st.session_state.custom_prompts = None

# Profanity warning system (3 strikes)
if "strike_count" not in st.session_state:
    st.session_state.strike_count = 0  # Track user warnings

def callback(indata, frames, time, status):
    """Callback function to store audio data"""
    if status:
        print(status)
    q.put(indata.copy())

# ✅ Define OpenAI Moderation API function
def check_profanity(text):
    """Check for inappropriate content using OpenAI Moderation API."""
    response = openai.Moderation.create(  # ✅ Fix: Call OpenAI directly
        input=text, 
        model="text-moderation-latest"
    )
    result = response["results"][0]
    flagged = result["flagged"]  # ✅ Fix: Use dictionary-style access

    # ✅ Fix: Handle categories correctly
    categories_dict = result["categories"]
    flagged_categories = [category for category, score in categories_dict.items() if score is not None and score > 0.5]

    return flagged, flagged_categories, []  # ✅ Return corrected data

# ✅ Define alternative response suggestion
def suggest_better_response(user_input):
    """Use AI to suggest a better, appropriate response instead of blocking."""
    prompt = f"사용자가 부적절한 내용을 입력했습니다: '{user_input}'. 이를 정중하게 바꾸고, 대화에 적절한 방식으로 다시 표현해주세요."
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

# ✅ Define chatbot response function
def chatbot_response(conversation_history):
    """Generate chatbot response using OpenAI GPT-4"""
    print("📡 Sending message history to GPT-4:", conversation_history)  # Debugging

    response = openai.ChatCompletion.create(  # ✅ Fix: Call OpenAI directly
        model="gpt-4-turbo",
        messages=conversation_history
    )

    chatbot_reply = response["choices"][0]["message"]["content"]
    print("🤖 GPT-4 Response:", chatbot_reply)  # Debugging
    return chatbot_reply

def record_audio(duration=18, samplerate=16000):
    """Records audio from the microphone and saves it as a WAV file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        filename = tmpfile.name
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(samplerate)

            print("Recording...")
            with sd.InputStream(callback=callback, samplerate=samplerate, channels=1, dtype="int16"):
                for _ in range(int(samplerate / 1024 * duration)):
                    wf.writeframes(q.get())

            print("Recording complete.")
    return filename

def transcribe_audio_whisper_api(audio_path):
    """Send audio file to OpenAI Whisper API and return transcribed text"""
    with open(audio_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text

# ✅ Define Whisper TTS function
def whisper_tts(text):
    """Convert chatbot response to speech using OpenAI's TTS API"""
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        # ✅ Ensure the file is written correctly
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

        with open(audio_path, "wb") as audio_file:
            audio_file.write(response.content)

        return audio_path  # ✅ Return the correct file path

    except Exception as e:
        st.write(f"🚨 **TTS Error:** {e}")
        return None
    
    # Save the audio response
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(audio_path, "wb") as audio_file:
        audio_file.write(response.content)

    return audio_path

# Collect personal information
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "custom_prompts" not in st.session_state:
    st.session_state.custom_prompts = None

st.title("🗣️ 맞춤형 한국어 대화 튜터")

# **Step 1: User Info Collection**
if st.session_state.user_info is None:
    st.write("### 📋 한국어 대화를 개인 맞춤형으로 설정하기 위해 정보를 입력해주세요.")
    
    with st.form("user_info_form"):
        name = st.text_input("이름")
        nationality = st.text_input("국적")
        native_language = st.text_input("모국어")
        
        residence_status = st.radio("대한민국 체류/거주 여부", ["네", "아니요"])

        # ✅ If "네" is selected, ask about visa details
        if residence_status == "네":
            stay_duration = st.text_input("한국 체류기간 (예: 1년, 6개월)")

            # ✅ Visa Type Dropdown
            visa_options = ["없음","C4", "D2", "D3", "D4", "D10", "E4", "E7", "E8", "E9",
                            "H2", "F1", "F2", "F3", "F4", "F6", "G1", "기타(직접입력)"]

            visa_type = st.selectbox("비자 종류를 선택하세요:", visa_options)

            # ✅ If "기타(직접입력)" is selected, allow manual input
            if visa_type == "기타(직접입력)":
                visa_type = st.text_input("비자 종류를 직접 입력하세요:")
        else:
            stay_duration = "해당 없음"
            visa_type = "해당 없음"

        industry = st.text_input("산업 분야 (예: IT, 교육, 의료 등)")
        work_experience = st.text_input("위 산업 분야 근무 기간")
        korean_test_score = st.text_input("한국어 시험 점수 (본 적 없으면 공란)")
        korean_study_duration = st.text_input("한국어 공부 기간 (예: 2년)")
        interests = st.text_input("관심 분야 (예: 여행, 역사, 음식)")
        hobbies = st.text_input("취미 (예: 축구, 독서, 게임)")
        
        agree = st.checkbox("📜 개인정보 수집 동의: 이 정보는 개인 맞춤형 대화 주제를 생성하는 데만 사용됩니다.", value=True)
        
        submitted = st.form_submit_button("제출")

    if submitted and agree:
        st.session_state.user_info = {
            "이름": name,
            "국적": nationality,
            "모국어": native_language,
            "대한민국 체류 여부": residence_status,
            "체류 기간": stay_duration,
            "비자 종류": visa_type,
            "산업 분야": industry,
            "근무 기간": work_experience,
            "한국어 시험 점수": korean_test_score,
            "한국어 공부 기간": korean_study_duration,
            "관심 분야": interests,
            "취미": hobbies,
        }
        st.rerun()

# **Step 2: Generate Personalized Prompts**
if st.session_state.user_info and st.session_state.custom_prompts is None:
    st.write("🤖 개인 맞춤형 대화 주제를 생성 중...")

    user_info_text = "\n".join([f"{k}: {v}" for k, v in st.session_state.user_info.items()])

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt_text}]
    )
    custom_prompts = []
    prompt_starters = []

    for line in response.choices[0].message.content.split("\n"):
        if ":" in line and '"' in line:
            topic_part, starter_part = line.split(":", 1)  # Split only at the first colon
            topic = topic_part.replace("-", "").replace('"', '').strip()  # Remove `- "` and quotes
            starter = starter_part.replace('"', '').strip()  # Remove extra quotes
            
            custom_prompts.append(topic)
            prompt_starters.append(starter)

    st.session_state.custom_prompts = custom_prompts
    st.session_state.prompt_starters = prompt_starters  # Store starting sentences separately

    # ✅ Debugging: Show generated prompts and openings
    print("🛠️ **Debugging Info:**")
    print("📌 AI-Generated Prompts:", st.session_state.custom_prompts)
    print("📢 AI-Generated Opening Sentences:", st.session_state.prompt_starters)
    st.rerun()

    # ✅ Debugging: Show generated prompts and openings

# **Step 3: Select Conversation Topic**
if st.session_state.custom_prompts and st.session_state.prompt_starters:
    st.write("🎯 **대화 주제를 선택하세요**:")

    # ✅ Predefined prompts with corresponding opening sentences
    predefined_prompts = {
        "🛒 옷 고르고 사기": "안녕하세요 손님! 무슨 옷을 사고 싶으신가요?",
        "🗺️ 방향 묻기": "안녕하세요. 어디 가고 싶으신 곳 있나요?",
        "🎉 재미있는 이벤트에 대해서 말해보기": "어제 무슨 재미있는 일이 있었나요?"
    }

    # ✅ AI-generated prompts (ensures correct mapping)
    ai_prompts = {
        "🆕 " + st.session_state.custom_prompts[0]: st.session_state.prompt_starters[0],
        "🆕 " + st.session_state.custom_prompts[1]: st.session_state.prompt_starters[1]
    }

    # ✅ Merge predefined & AI-generated prompts
    prompts = {**predefined_prompts, **ai_prompts}

    # ✅ User selects a topic
    selected_prompt = st.selectbox("대화를 시작할 주제를 선택하세요:", list(prompts.keys()))

    if st.button("🔄 대화 시작하기"):
        #st.write(f"🎯 **Selected Prompt:** {selected_prompt}")

        # ✅ Get chatbot's first response (opening sentence)
        chatbot_opening = prompts[selected_prompt] if selected_prompt in prompts else "이 주제에 대해 이야기해볼까요?"

        #st.write(f"📢 **Chatbot Opening:** {chatbot_opening}")

        # ✅ Store conversation history
        st.session_state.conversation_history = [
            {"role": "system", "content": "당신은 친절한 한국어 대화 파트너입니다. "
                                          "실제 생활에서 자연스럽게 대화를 나누듯이 응답하세요. "
                                          "너무 형식적인 문어체가 아닌 구어체로 대답하세요. "
                                          "사용자가 대화에 참여하도록 격려하세요. "
                                          "답변은 2~3문장으로 짧고 명확하게 하세요."},
            {"role": "assistant", "content": chatbot_opening}
        ]


        tts_audio = whisper_tts(chatbot_opening)

        if tts_audio:
            st.audio(tts_audio, format="audio/mp3", autoplay=True)

            # ✅ Add a short delay before rerunning the UI
            time.sleep(1)  # Allow time for TTS to play
        else:
            st.write("🚨 **TTS Failed to Generate Audio!**")

        # ✅ Store message history
        st.session_state.last_played_message = chatbot_opening
        st.session_state.response_count = 0
        st.session_state.chat_active = True
        st.rerun()

# **Step 4: Conversation Mode **
if st.session_state.chat_active:
    st.write("💬 **대화 기록**:")

    # ✅ Initialize TTS playback control in session state
    if "tts_playback" not in st.session_state:
        st.session_state.tts_playback = False  # Default: No playback until chatbot speaks

    if "last_played_message" not in st.session_state:
        st.session_state.last_played_message = ""  # Stores the last chatbot message played in TTS


    # ✅ Only play TTS for the latest chatbot message to prevent replays
    for i, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(f"👤 **You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"🤖 **Chatbot:** {msg['content']}")

            # ✅ Only play TTS for the latest assistant message & prevent replay during recording
            if i == len(st.session_state.conversation_history) - 1 and st.session_state.tts_playback:
                if msg["content"] != st.session_state.last_played_message:
                    tts_audio = whisper_tts(msg["content"])
                    st.audio(tts_audio, format="audio/mp3", autoplay=True)

                    # ✅ Store the last played message to prevent replaying
                    st.session_state.last_played_message = msg["content"]

    st.write(f"⏳ **진행 상황:** {st.session_state.response_count + 1} / 5 회")

    if st.session_state.response_count < 5:
        if st.button("🎙️ 음성 녹음 시작 (15초)"):
            st.session_state.tts_playback = False  # Disable TTS while recording
            st.session_state.last_played_message = ""  # Reset last played message to avoid replays
            with st.spinner("🎤 녹음 중... 15초 동안 말해주세요."):
                recorded_audio_path = record_audio()  # Replace with actual recording function

            st.write("📡 텍스트로 변환 중...")
            korean_text = transcribe_audio_whisper_api(recorded_audio_path)

            # 🚨 Check for profanity & get flagged categories and words
            flagged, flagged_categories, flagged_words = check_profanity(korean_text)

            if flagged:
                st.session_state.strike_count += 1

                # 🚨 Show warning with flagged words & categories
                warning_message = f"⚠️ **경고!** 부적절한 표현이 감지되었습니다.\n"
                if flagged_words:
                    warning_message += f"🚨 감지된 단어: {', '.join(flagged_words)}\n"
                if flagged_categories:
                    warning_message += f"📌 감지된 유형: {', '.join(flagged_categories)}\n"

                warning_message += f"⚠️ 앞으로 {3 - st.session_state.strike_count}번 더 경고를 받으면 대화가 종료됩니다."
                st.warning(warning_message)

                if st.session_state.strike_count >= 3:
                    st.write("🚨 **부적절한 표현이 여러 번 감지되었습니다.**")
                    st.write("⏸️ **대화가 일시 중지되었습니다. 계속하려면 아래 버튼을 눌러주세요.**")

                    # Show a button to return to topic selection instead of abruptly ending
                    if st.button("🔄 새로운 대화 시작하기"):
                        st.session_state.chat_active = False
                        st.session_state.response_count = 0
                        st.session_state.conversation_history = []
                        st.session_state.custom_prompts = None
                        st.session_state.user_info = None
                        st.session_state.strike_count = 0  # Reset warnings
                        st.rerun()

                    # ✅ Stop further execution so that the user sees the button
                    st.stop()
                else:
                    alternative_response = suggest_better_response(korean_text)

                    # 🔄 Change the chatbot's system prompt based on strike count
                    if st.session_state.strike_count == 1:
                        new_system_prompt = "⚠️ 주의: 이 대화에서는 예의 바르고 적절한 표현을 사용해야 합니다. 부적절한 언어는 피해주세요."
                    elif st.session_state.strike_count == 2:
                        new_system_prompt = "🚨 경고: 부적절한 표현이 계속 감지되고 있습니다. 다음 경고가 발생하면 대화가 종료됩니다."

                    # Update conversation history with new system prompt
                    st.session_state.conversation_history.insert(0, {"role": "system", "content": new_system_prompt})

                    # Append modified user response
                    st.write(f"🔹 **추천 표현:** {alternative_response}")
                    st.session_state.conversation_history.append({"role": "user", "content": f"[수정됨] {alternative_response}"})
            else:
                # ✅ If no profanity, proceed normally
                st.write(f"👤 **You:** {korean_text}")
                st.session_state.conversation_history.append({"role": "user", "content": korean_text})

                chatbot_reply = chatbot_response(st.session_state.conversation_history)
                st.session_state.conversation_history.append({"role": "assistant", "content": chatbot_reply})

                st.session_state.tts_playback = True  # ✅ Enable TTS playback for chatbot response
                st.session_state.last_played_message = ""  # ✅ Reset last played message to ensure it plays once
                st.rerun()

            # 챗봇 다음 대화
            st.session_state.response_count += 1

            # ✅ Debugging: Ensure count updates correctly
            st.write(f"🧮 **현재 대화 횟수:** {st.session_state.response_count} / 5")

            # ✅ Only rerun when necessary
            if st.session_state.response_count < 5:
                st.rerun()

    else:
        st.write("🎉 **대화가 끝났어요! 5번의 대화를 완료했습니다.**")
        if st.button("🔄 새로운 대화 시작하기"):
            st.session_state.chat_active = False
            st.session_state.response_count = 0
            st.session_state.conversation_history = []
            st.session_state.custom_prompts = None
            st.session_state.user_info = None
            st.rerun()