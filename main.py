import streamlit as st
from google_auth_oauthlib.flow import Flow
import os
import pandas as pd
from datetime import datetime
import csv
import hashlib
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 VizAI by Ethicallogix Login",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- HIDE SIDEBAR + DEFAULT MENU ----------------


# ---------------- PAGE STYLE ----------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f9ff;
            padding-top: 50px;
        }
        h1 {
            color: #4a90e2;
            text-align: center;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SESSION STATE ----------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'processed_code' not in st.session_state:
    st.session_state.processed_code = None

# ---------------- OAUTH CONFIG ----------------
CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8501"]
    }
}

SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# Disable HTTPS requirement for local dev
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# ---------------- FILE PATHS ----------------
LOGIN_LOG_FILE = "login_log.csv"
USERS_FILE = "users.csv"


hide_style = """
<style>
    /* Hide sidebar */
    div[data-testid="stSidebar"] {display: none !important;}
    /* Hide top collapsed toggle */
    button[data-testid="collapsedControl"] {display: none !important;}
    /* Hide default menu, footer, and header */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    /* Optional: add padding for main container */
    .block-container {padding-top: 3rem;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)
# ---------------- HELPERS ----------------
# ---------------- DEFAULT ADMIN CREATION ----------------
USERS_FILE = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
def ensure_default_admin():
    """Create default admin account if users.csv missing or admin not found."""
    if not os.path.isfile(USERS_FILE):
        df = pd.DataFrame([{
            "username": "hasi",
            "email": "hasi@ethicallogix.com",
            "password": hash_password("system786@"),
            "full_name": "Admin User",
            "role": "admin",
            "status": "active",
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        df.to_csv(USERS_FILE, index=False)
    else:
        df = pd.read_csv(USERS_FILE, on_bad_lines="skip")
        if "hasi" not in df["username"].values:
            new_admin = {
                "username": "hasi",
                "email": "hasi@ethicallogix.com",
                "password": hash_password("system786@"),
                "full_name": "Admin User",
                "role": "admin",
                "status": "active",
                "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            df = pd.concat([df, pd.DataFrame([new_admin])], ignore_index=True)
            df.to_csv(USERS_FILE, index=False)

ensure_default_admin()

def save_login_to_csv(user_info, login_method):
    """Save successful logins"""
    try:
        login_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email': user_info.get('email', 'N/A'),
            'name': user_info.get('name', 'N/A'),
            'login_method': login_method,
            'login_status': 'Success'
        }

        file_exists = os.path.isfile(LOGIN_LOG_FILE)
        with open(LOGIN_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'email', 'name', 'login_method', 'login_status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(login_data)
        return True
    except Exception as e:
        st.error(f"Failed to save login log: {str(e)}")
        return False

def register_user(username, email, password, full_name, role='user'):
    """Register a new user"""
    try:
        file_exists = os.path.isfile(USERS_FILE)
        if file_exists:
            df = pd.read_csv(USERS_FILE)
            if username in df['username'].values:
                return False, "Username already exists!"
            if email in df['email'].values:
                return False, "Email already registered!"

        hashed_pw = hash_password(password)
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_pw,
            'full_name': full_name,
            'role': role,
            'status': 'active',
            'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(USERS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['username', 'email', 'password', 'full_name', 'role', 'status', 'registration_date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(user_data)

        return True, "Registration successful!"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate existing user"""
    try:
        if not os.path.isfile(USERS_FILE):
            return False, None

        df = pd.read_csv(USERS_FILE, on_bad_lines='skip')
        if not all(col in df.columns for col in ['username', 'password']):
            st.error("Users file corrupted. Please delete users.csv and register again.")
            return False, None

        user_row = df[df['username'] == username]
        if user_row.empty:
            return False, None

        if user_row.iloc[0]['password'] == hash_password(password):
            user_info = {
                'username': user_row.iloc[0]['username'],
                'email': user_row.iloc[0].get('email', username),
                'name': user_row.iloc[0].get('full_name', username),
                'role': user_row.iloc[0].get('role', 'user')
            }
            return True, user_info
        return False, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None

def get_or_create_google_user(user_info):
    """Get or create Google user"""
    try:
        file_exists = os.path.isfile(USERS_FILE)
        email = user_info.get('email')

        if file_exists:
            df = pd.read_csv(USERS_FILE, on_bad_lines='skip')
            if 'email' in df.columns:
                user_row = df[df['email'] == email]
                if not user_row.empty:
                    return {
                        'username': user_row.iloc[0]['username'],
                        'email': email,
                        'name': user_row.iloc[0]['full_name'],
                        'role': user_row.iloc[0]['role']
                    }

        username = email.split('@')[0]
        hashed_pw = hash_password(os.urandom(32).hex())
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_pw,
            'full_name': user_info.get('name', username),
            'role': 'user',
            'status': 'active',
            'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(USERS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['username', 'email', 'password', 'full_name', 'role', 'status', 'registration_date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(user_data)

        return {
            'username': username,
            'email': email,
            'name': user_info.get('name', username),
            'role': 'user'
        }
    except Exception as e:
        st.error(f"Google user error: {str(e)}")
        return None

# ---------------- OAUTH CALLBACK ----------------
query_params = st.query_params
if "code" in query_params and st.session_state.user is None:
    auth_code = query_params["code"]
    if auth_code != st.session_state.processed_code:
        with st.spinner("Authenticating with Google..."):
            try:
                flow = Flow.from_client_config(
                    CLIENT_CONFIG,
                    scopes=SCOPES,
                    redirect_uri='http://localhost:8501'
                )
                flow.fetch_token(code=auth_code)
                credentials = flow.credentials

                user_info_response = requests.get(
                    'https://www.googleapis.com/oauth2/v1/userinfo',
                    headers={'Authorization': f'Bearer {credentials.token}'}
                )

                if user_info_response.status_code == 200:
                    user_info = user_info_response.json()
                    st.session_state.user = get_or_create_google_user(user_info)
                    st.session_state.processed_code = auth_code
                    save_login_to_csv(user_info, 'Google OAuth')
                    st.query_params.clear()
                    st.rerun()

            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                st.session_state.processed_code = auth_code
                st.query_params.clear()

# ---------------- LOGIN / REGISTER UI ----------------
if st.session_state.user:
    st.switch_page("Pages/Home.py")
else:
    st.title("VizAI by Ethicallogix")

    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "🌐 Google Sign-in"])

    # LOGIN TAB
    with tab1:
        st.subheader("Login with Username")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login", use_container_width=True)

            if login_btn:
                if username and password:
                    success, user_info = authenticate_user(username, password)
                    if success:
                        st.session_state.user = user_info
                        save_login_to_csv(user_info, 'Username/Password')
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.warning("⚠️ Enter both username and password!")

    # REGISTER TAB
    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            reg_username = st.text_input("Username*")
            reg_email = st.text_input("Email*")
            reg_fullname = st.text_input("Full Name*")
            reg_password = st.text_input("Password*", type="password")
            reg_password_confirm = st.text_input("Confirm Password*", type="password")
            register_btn = st.form_submit_button("Register", use_container_width=True)

            if register_btn:
                if not all([reg_username, reg_email, reg_fullname, reg_password, reg_password_confirm]):
                    st.warning("⚠️ Fill all required fields!")
                elif reg_password != reg_password_confirm:
                    st.error("❌ Passwords do not match!")
                elif len(reg_password) < 6:
                    st.error("❌ Password must be at least 6 characters long!")
                else:
                    success, msg = register_user(reg_username, reg_email, reg_password, reg_fullname)
                    st.success(msg) if success else st.error(msg)

    # GOOGLE SIGN-IN TAB
    with tab3:
        st.subheader("Sign in with Google")
        flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES, redirect_uri='http://localhost:8501')
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        st.link_button("🔐 Sign in with Google", auth_url, use_container_width=True)
