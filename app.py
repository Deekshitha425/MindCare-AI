"""
MindCare AI — Streamlit Application Entry Point
Run with:  streamlit run app.py
"""
import os, sys
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="MindCare AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
css_path = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.auth_manager import signup, login, is_logged_in, logout

# ─────────────────────────────────────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if not is_logged_in():

    # Hide sidebar completely on login screen
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* ── Full-screen animated background ── */
    .auth-bg {
        position: fixed; inset: 0; z-index: 0;
        background: linear-gradient(135deg, #0A0F1E 0%, #0D1B3E 30%, #0E2759 60%, #1a3a6b 100%);
        overflow: hidden;
    }
    /* Floating orbs */
    .orb {
        position: absolute; border-radius: 50%;
        filter: blur(80px); opacity: 0.18; animation: drift linear infinite;
    }
    .orb1 { width:520px;height:520px; background:#4F8EF7; top:-120px; left:-100px; animation-duration:18s; }
    .orb2 { width:400px;height:400px; background:#7EC8E3; bottom:-80px; right:-80px; animation-duration:22s; animation-delay:-6s; }
    .orb3 { width:300px;height:300px; background:#A8E6CF; top:40%; left:55%; animation-duration:28s; animation-delay:-12s; }
    .orb4 { width:200px;height:200px; background:#6B9FF8; top:20%; right:20%; animation-duration:16s; animation-delay:-4s; }
    @keyframes drift {
        0%   { transform: translate(0,0) scale(1); }
        33%  { transform: translate(40px,-30px) scale(1.05); }
        66%  { transform: translate(-20px,40px) scale(0.95); }
        100% { transform: translate(0,0) scale(1); }
    }

    /* Stars */
    .stars { position:absolute; inset:0; }
    .star  {
        position:absolute; background:#fff; border-radius:50%;
        animation: twinkle ease-in-out infinite;
    }
    @keyframes twinkle {
        0%,100% { opacity:0.15; transform:scale(1); }
        50%      { opacity:0.8;  transform:scale(1.4); }
    }

    /* ── Card ── */
    .auth-card {
        position: relative; z-index: 10;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 24px;
        padding: 2.6rem 2.4rem 2.2rem;
        box-shadow: 0 24px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.12);
        animation: fadeUp 0.6s ease both;
    }
    @keyframes fadeUp {
        from { opacity:0; transform:translateY(28px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* Brain pulse */
    .brain-icon {
        font-size: 3.8rem; display: block; text-align: center;
        animation: pulse 2.6s ease-in-out infinite;
        filter: drop-shadow(0 0 18px rgba(79,142,247,0.7));
    }
    @keyframes pulse {
        0%,100% { transform:scale(1);   filter:drop-shadow(0 0 14px rgba(79,142,247,0.5)); }
        50%      { transform:scale(1.08);filter:drop-shadow(0 0 28px rgba(79,142,247,0.9)); }
    }

    /* Gradient headline */
    .auth-title {
        background: linear-gradient(135deg, #ffffff, #93C5FD, #7EC8E3);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem; font-weight: 800; text-align: center;
        letter-spacing: -0.02em; margin: 6px 0 2px;
    }
    .auth-sub {
        text-align:center; color:#93C5FD; font-size:0.83rem;
        letter-spacing:0.04em; margin-bottom: 1.8rem;
    }

    /* Tab buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s !important;
        padding: 0.55rem 1rem !important;
    }

    /* Inputs on auth page */
    .auth-page .stTextInput input {
        background: rgba(255,255,255,0.09) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: #fff !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.92rem !important;
        transition: border 0.2s, box-shadow 0.2s !important;
    }
    .auth-page .stTextInput input:focus {
        border-color: #4F8EF7 !important;
        box-shadow: 0 0 0 3px rgba(79,142,247,0.25) !important;
    }
    .auth-page .stTextInput label {
        color: #CBD5E1 !important; font-size: 0.82rem !important;
        font-weight: 600 !important; letter-spacing: 0.03em !important;
    }
    .auth-page .stTextInput input::placeholder { color: rgba(255,255,255,0.3) !important; }

    /* Feature pills under card */
    .feature-pills {
        display:flex; gap:10px; justify-content:center;
        flex-wrap:wrap; margin-top:2rem; position:relative; z-index:10;
    }
    .pill {
        background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12);
        border-radius:50px; padding:6px 16px;
        font-size:0.75rem; color:#93C5FD; font-weight:500;
        backdrop-filter:blur(8px);
    }

    /* Divider */
    .auth-divider {
        display:flex; align-items:center; gap:10px;
        margin:0.8rem 0; color:rgba(255,255,255,0.25); font-size:0.75rem;
    }
    .auth-divider::before,.auth-divider::after {
        content:''; flex:1; height:1px; background:rgba(255,255,255,0.12);
    }
    </style>

    <!-- Animated background -->
    <div class="auth-bg">
        <div class="orb orb1"></div>
        <div class="orb orb2"></div>
        <div class="orb orb3"></div>
        <div class="orb orb4"></div>
        <div class="stars" id="stars"></div>
    </div>

    <script>
    // Generate twinkling stars
    const container = document.getElementById('stars');
    if (container) {
        for (let i = 0; i < 90; i++) {
            const s = document.createElement('div');
            s.className = 'star';
            const sz = Math.random() * 2.4 + 0.6;
            s.style.cssText = `width:${sz}px;height:${sz}px;
                top:${Math.random()*100}%;left:${Math.random()*100}%;
                animation-duration:${2+Math.random()*4}s;
                animation-delay:${Math.random()*4}s;`;
            container.appendChild(s);
        }
    }
    </script>
    """, unsafe_allow_html=True)

    # ── Layout ────────────────────────────────────────────────────────────────
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown('<div class="auth-page">', unsafe_allow_html=True)

        # Card header
        st.markdown("""
        <div class="auth-card">
            <span class="brain-icon">🧠</span>
            <div class="auth-title">MindCare AI</div>
            <div class="auth-sub">✦ Mental Health Support System ✦</div>
        </div>
        """, unsafe_allow_html=True)

        # Tab toggle
        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "Login"

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔑  Login", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "Login" else "secondary"):
                st.session_state.auth_mode = "Login"; st.rerun()
        with t2:
            if st.button("📝  Sign Up", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "Signup" else "secondary"):
                st.session_state.auth_mode = "Signup"; st.rerun()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── LOGIN FORM ────────────────────────────────────────────────────────
        if st.session_state.auth_mode == "Login":
            st.markdown("""
            <div style='color:rgba(255,255,255,0.6);font-size:0.78rem;
                        text-align:center;margin-bottom:0.8rem;letter-spacing:0.06em;
                        text-transform:uppercase;font-weight:600;'>
                Welcome back
            </div>""", unsafe_allow_html=True)

            login_user = st.text_input("Username", key="login_user", placeholder="Enter your username")
            login_pass = st.text_input("Password", type="password", key="login_pass",
                                       placeholder="Enter your password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("🚀  Login to MindCare", use_container_width=True,
                         type="primary", key="do_login"):
                ok, msg = login(login_user, login_pass)
                if ok:
                    st.session_state.username = login_user.strip().lower().replace(" ","_")
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("""
            <div style='text-align:center;margin-top:0.8rem;font-size:0.75rem;color:#64748B;'>
                New here? Switch to <b style='color:#93C5FD;'>Sign Up</b> above
            </div>""", unsafe_allow_html=True)

        # ── SIGNUP FORM ───────────────────────────────────────────────────────
        else:
            st.markdown("""
            <div style='color:rgba(255,255,255,0.6);font-size:0.78rem;
                        text-align:center;margin-bottom:0.8rem;letter-spacing:0.06em;
                        text-transform:uppercase;font-weight:600;'>
                Create your account
            </div>""", unsafe_allow_html=True)

            su_user  = st.text_input("Username", key="su_user", placeholder="At least 3 characters")
            su_pass  = st.text_input("Password", type="password", key="su_pass",
                                     placeholder="At least 6 characters")
            su_pass2 = st.text_input("Confirm Password", type="password", key="su_pass2",
                                     placeholder="Repeat your password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("✨  Create My Account", use_container_width=True,
                         type="primary", key="do_signup"):
                if su_pass != su_pass2:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = signup(su_user, su_pass)
                    if ok:
                        st.success(f"🎉 {msg}")
                        st.info("Click **Login** tab above to sign in.")
                    else:
                        st.error(msg)

            st.markdown("""
            <div style='text-align:center;margin-top:0.8rem;font-size:0.75rem;color:#64748B;'>
                Already have an account? Switch to <b style='color:#93C5FD;'>Login</b> above
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)   # close auth-page div

        # Feature pills below card
        st.markdown("""
        <div class="feature-pills">
            <span class="pill">🔒 Private &amp; Secure</span>
            <span class="pill">🤖 AI-Powered</span>
            <span class="pill">📊 Depression Assessment</span>
            <span class="pill">💬 Emotion Chat</span>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP  (logged-in users only)
# ─────────────────────────────────────────────────────────────────────────────

# Force sidebar open and visible after login
st.markdown("""
<style>
/* Make sure sidebar is always visible after login */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    min-width: 260px !important;
    width: 260px !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem !important;
}
/* Sidebar button overrides - keep them styled nicely */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    text-align: left !important;
    padding: 0.5rem 0.9rem !important;
    transition: all 0.2s !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
    transform: translateX(3px) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton[data-testid*="logout"] > button,
[data-testid="stSidebar"] button[kind="secondary"] {
    background: rgba(239,68,68,0.15) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #FCA5A5 !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(239,68,68,0.28) !important;
}
</style>
""", unsafe_allow_html=True)

from components.sidebar import render_sidebar
from pages import home, chat, assessment, analytics, history, about


page = render_sidebar()

PAGE_MAP = {
    "Home":       home.render,
    "Chat":       chat.render,
    "Assessment": assessment.render,
    "Analytics":  analytics.render,
    "History":    history.render,
    "About":      about.render,
}

PAGE_MAP.get(page, home.render)()
