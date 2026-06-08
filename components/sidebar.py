"""
components/sidebar.py — Sidebar navigation with active-page highlight.
"""
import streamlit as st
from utils.auth_manager import logout

PAGES = {
    "🏠  Home":                   "Home",
    "💬  Emotion Support Chat":   "Chat",
    "📊  Depression Assessment":  "Assessment",
    "📈  Analytics Dashboard":    "Analytics",
    "📋  Assessment History":     "History",
}

PAGE_ICONS = {
    "Home": "🏠", "Chat": "💬", "Assessment": "📊",
    "Analytics": "📈", "History": "📋", "Profile": "👤",
}


def render_sidebar() -> str:
    # ── Force sidebar always visible ──────────────────────────────────────────
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        min-width: 270px !important;
        width: 270px !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg,#0D1B3E 0%,#1E3A5F 55%,#2D5A8E 100%) !important;
        min-height: 100vh !important;
    }
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        color: #4F8EF7 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    /* Navigation buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        color: #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.87rem !important;
        text-align: left !important;
        padding: 0.55rem 1rem !important;
        transition: all 0.2s !important;
        margin-bottom: 3px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(79,142,247,0.18) !important;
        border-color: rgba(79,142,247,0.35) !important;
        color: #fff !important;
        transform: translateX(3px) !important;
    }
    /* Active nav button */
    .nav-active > div > button,
    .nav-active button {
        background: linear-gradient(135deg,#4F8EF7,#6B9FF8) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79,142,247,0.4) !important;
        transform: translateX(4px) !important;
        color: #fff !important;
        font-weight: 700 !important;
    }
    /* Logout button */
    [data-testid="stSidebar"] .stButton[data-testid*="logout"] > button,
    [data-testid="stSidebar"] div[data-testid="element-container"]:last-of-type .stButton > button {
        background: rgba(239,68,68,0.12) !important;
        border-color: rgba(239,68,68,0.28) !important;
        color: #FCA5A5 !important;
    }
    [data-testid="stSidebar"] div[data-testid="element-container"]:last-of-type .stButton > button:hover {
        background: rgba(239,68,68,0.25) !important;
    }
    /* Hide default Streamlit page nav */
    [data-testid="stSidebarNav"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:

        # ── Brand ─────────────────────────────────────────────────────────────
        st.markdown("""
        <style>
        @keyframes pulse {
            0%,100% { filter: drop-shadow(0 0 10px rgba(79,142,247,0.7)); }
            50%      { filter: drop-shadow(0 0 20px rgba(79,142,247,1.0)); }
        }
        </style>
        <div style="text-align:center; padding:1.4rem 0 1rem;">
            <div style="font-size:2.6rem; animation:pulse 2.6s ease-in-out infinite;">🧠</div>
            <div style="font-size:1.1rem; font-weight:800; color:#fff;
                        letter-spacing:0.01em; margin-top:6px;">MindCare AI</div>
            <div style="font-size:0.72rem; color:#93C5FD; margin-top:2px;
                        letter-spacing:0.05em;">MENTAL HEALTH SUPPORT</div>
        </div>
        <hr style="border:none; border-top:1px solid rgba(255,255,255,0.12); margin:0 0 0.8rem;">
        """, unsafe_allow_html=True)

        # ── User badge ────────────────────────────────────────────────────────
        username = st.session_state.get("username", "guest")
        initials = username[:2].upper()
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px;
                    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12);
                    border-radius:12px; padding:9px 12px; margin-bottom:1.1rem;">
            <div style="width:32px; height:32px; border-radius:50%; flex-shrink:0;
                        background:linear-gradient(135deg,#4F8EF7,#7EC8E3);
                        display:flex; align-items:center; justify-content:center;
                        font-size:0.75rem; font-weight:800; color:#fff;">{initials}</div>
            <div>
                <div style="font-size:0.82rem; font-weight:700; color:#fff;">{username}</div>
                <div style="font-size:0.68rem; color:#93C5FD;">Active session</div>
            </div>
            <div style="margin-left:auto; width:7px; height:7px; border-radius:50%;
                        background:#22C55E; box-shadow:0 0 6px #22C55E;"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation label ──────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.68rem; color:#64748B; font-weight:700;
                    letter-spacing:0.1em; padding-left:4px; margin-bottom:6px;">
            NAVIGATION
        </div>""", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = "Home"

        current = st.session_state.page

        for label, page_key in PAGES.items():
            is_active = (current == page_key)
            if is_active:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            clicked = st.button(label, key=f"nav_{page_key}", use_container_width=True)
            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)
            if clicked:
                st.session_state.page = page_key
                st.rerun()

        # ── Divider + Logout ──────────────────────────────────────────────────
        st.markdown("""
        <hr style="border:none; border-top:1px solid rgba(255,255,255,0.10); margin:1rem 0 0.6rem;">
        """, unsafe_allow_html=True)

        if st.button("🚪  Logout", use_container_width=True, key="logout_btn"):
            logout()
            st.rerun()

        # ── Current page indicator ────────────────────────────────────────────
        icon = PAGE_ICONS.get(current, "📄")
        st.markdown(f"""
        <div style="margin-top:1.2rem; background:rgba(79,142,247,0.12);
                    border:1px solid rgba(79,142,247,0.25); border-radius:10px;
                    padding:8px 12px; text-align:center;">
            <div style="font-size:0.68rem; color:#93C5FD; font-weight:600;
                        letter-spacing:0.06em; text-transform:uppercase;">Now Viewing</div>
            <div style="font-size:0.88rem; color:#fff; font-weight:700; margin-top:2px;">
                {icon} {current}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.68rem; color:#475569; line-height:1.55;
                    padding:10px 4px 0; margin-top:0.8rem; border-top:1px solid rgba(255,255,255,0.07);">
            ⚠️ For self-awareness only. Not a substitute for professional medical advice.
        </div>
        """, unsafe_allow_html=True)

    return st.session_state.page
