import streamlit as st
import pandas as pd
import os
from datetime import datetime
import csv
import hashlib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🔧 Admin Portal",
    page_icon="🔧",
    layout="wide"
)

# ---------------- ACCESS CONTROL ----------------
if not st.session_state.get('user'):
    st.switch_page("main.py")
elif st.session_state.user.get('role') != 'admin':
    st.error("⛔ Access Denied! You must be an admin to access this page.")
    st.info("Redirecting to home...")
    st.switch_page("pages/Home.py")
else:
    user = st.session_state.user

    # ---------------- FILE PATHS ----------------
    USERS_FILE = "users.csv"
    LOGIN_LOG_FILE = "login_log.csv"

    # ---------------- HELPERS ----------------
    def hash_password(password):
        """Hash password securely."""
        return hashlib.sha256(password.encode()).hexdigest()

    def load_users():
        """Load user data."""
        if os.path.isfile(USERS_FILE):
            return pd.read_csv(USERS_FILE)
        return pd.DataFrame(columns=['username', 'email', 'password', 'full_name', 'role', 'status', 'registration_date'])


    def load_login_logs():
        """Load login logs safely, even if file has malformed rows."""
        if os.path.isfile(LOGIN_LOG_FILE):
            try:
                return pd.read_csv(LOGIN_LOG_FILE, on_bad_lines='skip', engine='python')
            except Exception as e:
                st.warning(f"⚠️ Login log file corrupted: {e}. Reinitializing...")
                os.rename(LOGIN_LOG_FILE, f"{LOGIN_LOG_FILE}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                return pd.DataFrame(columns=['timestamp', 'email', 'name', 'login_method', 'login_status'])
        return pd.DataFrame(columns=['timestamp', 'email', 'name', 'login_method', 'login_status'])

    def save_users(df):
        """Save users to CSV."""
        df.to_csv(USERS_FILE, index=False)

    def delete_user(username):
        """Delete a specific user."""
        df = load_users()
        df = df[df['username'] != username]
        save_users(df)

    def update_user_role(username, new_role):
        """Update user role."""
        df = load_users()
        df.loc[df['username'] == username, 'role'] = new_role
        save_users(df)

    def update_user_status(username, new_status):
        """Update user status."""
        df = load_users()
        df.loc[df['username'] == username, 'status'] = new_status
        save_users(df)

    def add_user(username, email, password, full_name, role, status='active'):
        """Add a new user."""
        df = load_users()

        if username in df['username'].values:
            return False, "Username already exists!"
        if email in df['email'].values:
            return False, "Email already registered!"

        hashed_pw = hash_password(password)
        new_user = pd.DataFrame([{
            'username': username,
            'email': email,
            'password': hashed_pw,
            'full_name': full_name,
            'role': role,
            'status': status,
            'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])

        df = pd.concat([df, new_user], ignore_index=True)
        save_users(df)
        return True, "User added successfully!"

    # ---------------- PAGE STYLE ----------------
    st.markdown("""
        <style>
            .main {
                background-color: #f4f9ff;
            }
            .admin-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
            /* 🔒 Hide Streamlit's built-in sidebar and navigation */
            [data-testid="stSidebar"] {display: none;}
            [data-testid="stSidebarNav"] {display: none;}
            [data-testid="collapsedControl"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("👤 Admin Profile")

        if user.get('picture'):
            st.image(user.get('picture'), width=100)
        else:
            st.markdown("<h1 style='text-align:center;'>🔧</h1>", unsafe_allow_html=True)

        st.write(f"**Username:** {user.get('username')}")
        st.write(f"**Email:** {user.get('email')}")
        st.write(f"**Role:** 🔧 Admin")

        st.divider()

        st.header("📋 Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("pages/Home.py")

        if st.button("🔧 Admin Portal", use_container_width=True, type="primary"):
            st.rerun()

        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.switch_page("main.py")

    # ---------------- HEADER ----------------
    st.markdown("""
        <div class='admin-header'>
            <h1>🔧 Admin Portal</h1>
            <p>System Management Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    # ---------------- LOAD DATA ----------------
    users_df = load_users()
    login_logs_df = load_login_logs()

    # ---------------- METRICS ----------------
    st.header("📊 System Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(users_df))
    with col2:
        st.metric("Admin Users", len(users_df[users_df['role'] == 'admin']))
    with col3:
        st.metric("Active Users", len(users_df[users_df['status'] == 'active']))
    with col4:
        st.metric("Total Logins", len(login_logs_df))

    st.divider()

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "➕ Add User", "📋 Login Logs", "⚙️ Settings"])

    # TAB 1 — USER MANAGEMENT
    with tab1:
        st.subheader("👥 Manage Users")

        if len(users_df) == 0:
            st.info("No users found.")
        else:
            for idx, row in users_df.iterrows():
                with st.expander(f"👤 {row['username']} — {row['email']}", expanded=False):
                    col_info, col_actions = st.columns([2, 1])

                    with col_info:
                        st.write(f"**Full Name:** {row['full_name']}")
                        st.write(f"**Role:** {row['role']}")
                        st.write(f"**Status:** {row['status']}")
                        st.write(f"**Registered:** {row['registration_date']}")

                    with col_actions:
                        st.write("**Actions**")

                        new_role = st.selectbox(
                            "Change Role", ["user", "admin"],
                            index=0 if row['role'] == 'user' else 1,
                            key=f"role_{row['username']}"
                        )
                        if st.button("Update Role", key=f"update_role_{row['username']}"):
                            if row['username'] == user.get('username'):
                                st.warning("⚠️ Cannot change your own role!")
                            else:
                                update_user_role(row['username'], new_role)
                                st.success(f"✅ Role updated to {new_role}")
                                st.rerun()

                        new_status = st.selectbox(
                            "Change Status", ["active", "inactive"],
                            index=0 if row['status'] == 'active' else 1,
                            key=f"status_{row['username']}"
                        )
                        if st.button("Update Status", key=f"update_status_{row['username']}"):
                            if row['username'] == user.get('username'):
                                st.warning("⚠️ Cannot change your own status!")
                            else:
                                update_user_status(row['username'], new_status)
                                st.success(f"✅ Status updated to {new_status}")
                                st.rerun()

                        if st.button("🗑️ Delete User", key=f"delete_{row['username']}"):
                            if row['username'] == user.get('username'):
                                st.error("❌ Cannot delete your own account!")
                            else:
                                delete_user(row['username'])
                                st.success(f"✅ User {row['username']} deleted!")
                                st.rerun()

    # TAB 2 — ADD USER
    with tab2:
        st.subheader("➕ Add New User")

        with st.form("add_user_form"):
            uname = st.text_input("Username*")
            email = st.text_input("Email*")
            fullname = st.text_input("Full Name*")
            password = st.text_input("Password*", type="password")
            role = st.selectbox("Role*", ["user", "admin"])
            status = st.selectbox("Status*", ["active", "inactive"])
            submitted = st.form_submit_button("✅ Add User", use_container_width=True)

            if submitted:
                if not all([uname, email, fullname, password]):
                    st.warning("⚠️ Please fill all fields.")
                elif '@' not in email:
                    st.error("❌ Enter a valid email.")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters.")
                else:
                    success, msg = add_user(uname, email, password, fullname, role, status)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # TAB 3 — LOGIN LOGS
    with tab3:
        st.subheader("📋 Login Activity Logs")

        if len(login_logs_df) == 0:
            st.info("No login logs found.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                method_filter = st.multiselect(
                    "Filter by Method",
                    login_logs_df['login_method'].unique(),
                    default=login_logs_df['login_method'].unique()
                )
            with c2:
                status_filter = st.multiselect(
                    "Filter by Status",
                    login_logs_df['login_status'].unique(),
                    default=login_logs_df['login_status'].unique()
                )

            filtered = login_logs_df[
                (login_logs_df['login_method'].isin(method_filter)) &
                (login_logs_df['login_status'].isin(status_filter))
            ].sort_values('timestamp', ascending=False)

            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Logs",
                data=filtered.to_csv(index=False),
                file_name=f"login_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # TAB 4 — SETTINGS
    with tab4:
        st.subheader("⚙️ System Settings")
        st.warning("🚧 Settings panel coming soon!")
        st.write("- Configure system parameters")
        st.write("- Email notifications")
        st.write("- Security policies")
        st.write("- Backup and restore")

        st.divider()
        st.subheader("🗄️ Database Files")

        colA, colB = st.columns(2)
        with colA:
            if os.path.isfile(USERS_FILE):
                st.success(f"✅ Users DB found ({USERS_FILE})")
                st.caption(f"Size: {os.path.getsize(USERS_FILE)} bytes")
            else:
                st.error("❌ Users file missing!")

        with colB:
            if os.path.isfile(LOGIN_LOG_FILE):
                st.success(f"✅ Login logs found ({LOGIN_LOG_FILE})")
                st.caption(f"Size: {os.path.getsize(LOGIN_LOG_FILE)} bytes")
            else:
                st.error("❌ Login log file missing!")
