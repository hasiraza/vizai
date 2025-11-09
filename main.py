import streamlit as st
import pandas as pd
import os
from datetime import datetime
import csv
import hashlib
import requests
from google_auth_oauthlib.flow import Flow

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 VizAI by Ethicallogix Login",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'processed_code' not in st.session_state:
    st.session_state.processed_code = None

# ---------------- FILE PATHS ----------------
USERS_FILE = "users.csv"
LOGIN_LOG_FILE = "login_log.csv"

# ---------------- HELPERS ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def ensure_default_admin():
    """Create default admin if not exists"""
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

ensure_default_admin()

def save_login_to_csv(user_info, login_method):
    login_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'email': user_info.get('email', 'N/A'),
        'name': user_info.get('name', 'N/A'),
        'login_method': login_method,
        'login_status': 'Success'
    }
    file_exists = os.path.isfile(LOGIN_LOG_FILE)
    with open(LOGIN_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=login_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(login_data)

def authenticate_user(username, password):
    if not os.path.isfile(USERS_FILE):
        return False, None
    df = pd.read_csv(USERS_FILE, on_bad_lines='skip')
    user_row = df[df['username'] == username]
    if not user_row.empty and user_row.iloc[0]['password'] == hash_password(password):
        return True, {
            'username': user_row.iloc[0]['username'],
            'email': user_row.iloc[0]['email'],
            'name': user_row.iloc[0]['full_name'],
            'role': user_row.iloc[0]['role']
        }
    return False, None

def register_user(username, email, password, full_name):
    df = pd.read_csv(USERS_FILE) if os.path.isfile(USERS_FILE) else pd.DataFrame()
    if username in df.get('username', []):
        return False, "Username already exists!"
    if email in df.get('email', []):
        return False, "Email already registered!"
    hashed_pw = hash_password(password)
    new_user = {
        'username': username,
        'email': email,
        'password': hashed_pw,
        'full_name': full_name,
        'role': 'user',
        'status': 'active',
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True, "Registration successful!"

# ---------------- OAUTH CONFIG ----------------
CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["https://your-app-url.streamlit.app"]
    }
}
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def get_or_create_google_user(user_info):
    email = user_info.get('email')
    df = pd.read_csv(USERS_FILE) if os.path.isfile(USERS_FILE) else pd.DataFrame()
    user_row = df[df.get('email', '') == email]
    if not user_row.empty:
        return {
            'username': user_row.iloc[0]['username'],
            'email': email,
            'name': user_row.iloc[0]['full_name'],
            'role': user_row.iloc[0]['role']
        }
    # create new user
    username = email.split("@")[0]
    hashed_pw = hash_password(os.urandom(32).hex())
    new_user = {
        'username': username,
        'email': email,
        'password': hashed_pw,
        'full_name': user_info.get('name', username),
        'role': 'user',
        'status': 'active',
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return new_user

# ---------------- LOGIN / REGISTER UI ----------------
st.title("VizAI by Ethicallogix")

tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

with tab1:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        success, user_info = authenticate_user(username, password)
        if success:
            st.session_state.user = user_info
            save_login_to_csv(user_info, "Username/Password")
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid credentials!")

with tab2:
    st.subheader("Register")
    reg_username = st.text_input("New Username")
    reg_email = st.text_input("Email")
    reg_name = st.text_input("Full Name")
    reg_password = st.text_input("Password", type="password")
    reg_confirm = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if reg_password != reg_confirm:
            st.error("❌ Passwords do not match!")
        else:
            success, msg = register_user(reg_username, reg_email, reg_password, reg_name)
            if success:
                st.success(msg)
            else:
                st.error(msg)

# Redirect to Home page if logged in
if st.session_state.user:
    st.experimental_set_query_params()  # clear query params
    st.switch_page("Home")
