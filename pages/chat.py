"""
pages/chat.py — Emotion Support Chat
Uses local emotion model + keyword fallback chatbot logic. No external APIs.
"""
import datetime
import random
import streamlit as st
from components.cards import section_header, emotion_badge
from utils.data_loader import load_emotion_model, predict_emotion
from utils.history_manager import load_conversation_history, save_conversation_turn

# ── Confidence threshold ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60

# ── Keyword fallback ──────────────────────────────────────────────────────────
KEYWORD_EMOTION_MAP = [
    (["tired","exhausted","drained","worn out","fatigued","no energy",
      "burnout","can't sleep","sleepless","heavy"], "tired"),
    (["lonely","alone","no one","nobody","isolated","no friends",
      "dont have anyone","don't have anyone","no one to talk",
      "no one cares","left out","invisible"], "lonely"),
    (["sad","unhappy","miserable","crying","cry","hopeless",
      "empty","broken","grief","loss"], "sadness"),
    (["anxious","anxiety","worried","worry","scared","nervous",
      "panic","overwhelmed","stressed","can't cope"], "fear"),
    (["angry","frustrated","mad","annoyed","furious","irritated","rage"], "anger"),
    (["happy","excited","wonderful","joyful","grateful","fantastic"], "joy"),
]

def _keyword_emotion(text: str) -> str:
    lowered = text.lower()
    for keywords, emotion in KEYWORD_EMOTION_MAP:
        if any(kw in lowered for kw in keywords):
            return emotion
    return "neutral"

# ── Response templates ────────────────────────────────────────────────────────
RESPONSES = {
    "tired": [
        "I hear you — feeling tired and drained is really hard. What's been taking the most out of you lately?",
        "Exhaustion can make everything feel so much heavier. Have you been able to get any proper rest?",
        "That kind of deep tiredness often signals that something needs to change. Is it physical exhaustion, emotional exhaustion, or both?",
        "Being constantly tired affects everything — your mood, your focus, your motivation. What do you think has been wearing you down?",
    ],
    "lonely": [
        "I'm really glad you reached out. Feeling like you have no one to talk to is one of the hardest feelings. I'm here — what's been on your mind?",
        "Loneliness can be incredibly painful. You don't have to face it alone right now. Would you like to tell me what's been going on?",
        "It takes courage to reach out when you're feeling isolated. What's made it hard to connect with people lately?",
        "I'm here and I'm listening. Sometimes just having someone to talk to makes a difference. What would you like to share?",
    ],
    "sadness": [
        "I'm sorry you're feeling this way. What's been weighing on your heart lately?",
        "It's okay to feel sad — you don't have to push through it alone. Can you tell me more about what's been happening?",
        "I hear that you're hurting. Would you like to talk about what's been going on?",
        "Thank you for sharing that with me. What do you think has been the hardest part?",
    ],
    "fear": [
        "It sounds like you've been carrying a lot of worry. What's been feeling most overwhelming right now?",
        "Anxiety can be exhausting to live with. What's been on your mind the most?",
        "When we're stressed or scared, everything can feel out of control. What feels most uncertain for you right now?",
        "I hear you. What do you think is at the root of what's been stressing you out?",
    ],
    "anger": [
        "It sounds like something has really been frustrating you. What's been building up?",
        "Anger often signals that something important feels unfair or threatened. What's been happening?",
        "I can hear that you're frustrated. Want to talk about what's been going on?",
    ],
    "joy": [
        "That's really good to hear! What's been making things feel positive?",
        "I'm glad you're feeling good. What's been going well for you?",
        "It's wonderful to hear some positivity! What happened?",
    ],
    "neutral": [
        "I'm here and happy to listen. What's been on your mind?",
        "What would you like to talk about today?",
        "Tell me what's going on — I'm here for you.",
    ],
    "surprise": [
        "That sounds unexpected! How are you processing what happened?",
        "Life can throw us off sometimes. What's caught you off guard?",
    ],
    "love": [
        "It sounds like you're feeling connected. Who or what has been a source of warmth for you?",
        "That's a lovely feeling to carry. Tell me more.",
    ],
}

FOLLOW_UPS = [
    "How long have you been feeling this way?",
    "Is there anyone in your life you feel comfortable talking to about this?",
    "How has your sleep been lately?",
    "What does a typical day look like for you right now?",
    "Have you been able to do things that usually bring you some relief?",
]

CRISIS_KEYWORDS = ["suicid","kill myself","end my life","want to die",
                   "no reason to live","hurt myself","self harm","can't go on"]

def _is_crisis(text: str) -> bool:
    return any(kw in text.lower() for kw in CRISIS_KEYWORDS)

def _detect_intent(text: str) -> str:
    l = text.lower()
    if any(w in l for w in ["hello","hi","hey","good morning","good evening"]): return "greeting"
    if any(w in l for w in ["bye","goodbye","exit","quit","take care"]): return "farewell"
    if any(w in l for w in ["thank","thanks","appreciate"]): return "thanks"
    if any(w in l for w in ["what should i do","advice","recommend","suggest","tips"]): return "recommendation"
    return "general"

def _get_response(emotion: str, turn_count: int) -> str:
    pool   = RESPONSES.get(emotion, RESPONSES["neutral"])
    recent = st.session_state.get("recent_responses", [])
    unused = [r for r in pool if r not in recent]
    chosen = random.choice(unused if unused else pool)

    if "recent_responses" not in st.session_state:
        st.session_state.recent_responses = []
    st.session_state.recent_responses = (st.session_state.recent_responses + [chosen])[-15:]

    if turn_count > 0 and turn_count % 3 == 0:
        fq = random.choice(FOLLOW_UPS)
        chosen = chosen.rstrip() + "\n\n" + fq

    return chosen

def _generate_response(text: str, emotion: str, turn_count: int) -> str:
    intent = _detect_intent(text)

    if _is_crisis(text):
        return ("I'm very concerned about what you've shared. Please reach out immediately:\n\n"
                "📞 **iCall (India): 9152987821**\n"
                "📞 **Vandrevala Foundation: 1860-2662-345** (24/7)\n\n"
                "You matter. Is there someone who can be with you right now?")
    if intent == "greeting":
        return random.choice([
            "Hello! I'm MindCare AI. I'm here to listen. How are you feeling today?",
            "Hi there! I'm glad you're here. What's on your mind?",
            "Hey! How are you doing today?",
        ])
    if intent == "farewell":
        return "Take care of yourself. I'm here whenever you need to talk. 💙"
    if intent == "thanks":
        return "You're welcome. I'm always here if you need to talk more. 😊"
    if intent == "recommendation":
        return ("Here are some things that might help:\n\n"
                "• Try a 5-minute breathing exercise: 4 counts in, hold 4, exhale 6.\n"
                "• Reach out to one person today, even just a short message.\n"
                "• Consider speaking with a counsellor if feelings persist.\n\n"
                "Would you like to talk through any of these?")

    return _get_response(emotion, turn_count)


# ── Emotion colors ────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "joy":"#F59E0B","sadness":"#3B82F6","anger":"#EF4444","fear":"#8B5CF6",
    "love":"#EC4899","surprise":"#F97316","neutral":"#94A3B8",
    "tired":"#06B6D4","lonely":"#22C55E",
}


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    section_header("💬 Emotion Support Chat")
    st.markdown(
        '<p style="color:#64748B; font-size:0.9rem; margin-top:-0.5rem; margin-bottom:1.2rem;">'
        'Share how you\'re feeling. I\'m here to listen and support you — no judgement, no data shared externally.'
        '</p>', unsafe_allow_html=True
    )

    # ── Init session state ────────────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        # Load persisted history
        history = load_conversation_history()
        st.session_state.chat_messages = [
            {"role": "bot", "text": "Hello! I'm MindCare AI. I'm here to listen. How are you feeling today?",
             "emotion": None, "confidence": None, "ts": ""}
        ]
        if history:
            for rec in history[-20:]:   # load last 20 turns
                st.session_state.chat_messages.append(
                    {"role": "user", "text": rec.get("user",""), "emotion": None, "confidence": None,
                     "ts": rec.get("timestamp","")[:16].replace("T"," ")})
                em = rec.get("emotion", {})
                st.session_state.chat_messages.append(
                    {"role": "bot", "text": rec.get("bot",""),
                     "emotion": em.get("emotion") if isinstance(em, dict) else em,
                     "confidence": em.get("confidence") if isinstance(em, dict) else None,
                     "ts": rec.get("timestamp","")[:16].replace("T"," ")})

    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([6, 1])
    with ctrl_col2:
        if st.button("🗑️ Clear", help="Clear chat history"):
            st.session_state.chat_messages = [
                {"role": "bot",
                 "text": "Chat cleared. How are you feeling today?",
                 "emotion": None, "confidence": None, "ts": ""}
            ]
            st.session_state.turn_count = 0
            from utils.history_manager import clear_conversation_history
            clear_conversation_history()
            st.rerun()

    # ── Chat display ──────────────────────────────────────────────────────────
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.chat_messages:
        ts   = msg.get("ts","")
        text = msg["text"].replace("\n", "<br>")

        if msg["role"] == "user":
            chat_html += f'''
            <div style="display:flex; flex-direction:column; align-items:flex-end;">
                <div class="chat-user">{text}</div>
                <div class="chat-meta" style="text-align:right;">{ts}</div>
            </div>'''
        else:
            em   = msg.get("emotion")
            conf = msg.get("confidence")
            badge = ""
            if em:
                badge = f'<br><div style="margin-top:6px;">{emotion_badge(em)}'
                if conf:
                    col = EMOTION_COLORS.get(em, "#94A3B8")
                    pct = int(conf * 100)
                    badge += (f' <span style="font-size:0.72rem; color:{col}; '
                              f'font-weight:600; margin-left:6px;">{pct}% confidence</span>')
                badge += '</div>'
            chat_html += f'''
            <div style="display:flex; flex-direction:column; align-items:flex-start;">
                <div class="chat-bot">{text}{badge}</div>
                <div class="chat-meta">{ts}</div>
            </div>'''

    chat_html += '</div>'
    st.markdown(f'<div class="chat-scroll">{chat_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input(
                "Your message",
                placeholder="Type how you're feeling… (e.g. 'I feel really tired and overwhelmed')",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        text = user_input.strip()
        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Append user message
        st.session_state.chat_messages.append(
            {"role": "user", "text": text, "emotion": None, "confidence": None, "ts": ts}
        )

        # Emotion detection with confidence guard
        model = load_emotion_model()
        emotion, confidence, source = "neutral", 0.0, "fallback"
        if model:
            result    = predict_emotion(text, model)
            ml_em     = result["emotion"]
            ml_conf   = result["confidence"]
            if ml_conf >= CONFIDENCE_THRESHOLD:
                emotion, confidence, source = ml_em, ml_conf, "model"
            else:
                emotion    = _keyword_emotion(text)
                confidence = ml_conf
                source     = "keyword_fallback"
        else:
            emotion = _keyword_emotion(text)

        # Generate response
        st.session_state.turn_count += 1
        response = _generate_response(text, emotion, st.session_state.turn_count)

        st.session_state.chat_messages.append(
            {"role": "bot", "text": response, "emotion": emotion,
             "confidence": confidence, "ts": ts}
        )

        # Persist
        save_conversation_turn({
            "turn": st.session_state.turn_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "user": text, "bot": response,
            "emotion": {"emotion": emotion, "confidence": confidence, "source": source},
            "intent": _detect_intent(text),
            "risk_level": None,
        })

        st.rerun()

    # ── Tip ───────────────────────────────────────────────────────────────────
    username = st.session_state.get("username", "guest")
    st.markdown(f"""
    <div class="disclaimer" style="margin-top:0.5rem;">
        💡 <b>Tip:</b> This chat uses a local AI model trained on emotional language.
        No messages are sent to any external service.
        Conversations are saved privately for <b>{username}</b>.
    </div>
    """, unsafe_allow_html=True)
