import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer,
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK data quietly at the start
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="🧮 VizAi Data Cleaning", layout="wide")
st.title("🧮 VizAi Data Cleaning - Ethicallogix")

# ----------------- HELP FUNCTION -----------------
def show_help(tab_key, title, message):
    """Display help popup for a specific tab."""
    help_flag = f"show_help_{tab_key}"
    if help_flag not in st.session_state:
        st.session_state[help_flag] = False

    col_help = st.columns([8, 1])[1]
    with col_help:
        if st.button("❓ Help", key=f"help_btn_{tab_key}"):
            st.session_state[help_flag] = True
            st.rerun()

    if st.session_state[help_flag]:
        with st.expander(f"💡 {title}", expanded=True):
            st.markdown(message, unsafe_allow_html=True)
            st.button("❌ Close Help", key=f"close_help_{tab_key}",
                      on_click=lambda: st.session_state.update({help_flag: False}))

# ----------------- SIDEBAR -----------------
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type=["csv"])
mode = st.sidebar.radio("Choose Mode", ["Preview", "Download"])

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None
if "original_df" not in st.session_state:
    st.session_state["original_df"] = None
if "updated_df" not in st.session_state:
    st.session_state["updated_df"] = None

# ----------------- FILE HANDLING -----------------
if uploaded_file:
    filename = uploaded_file.name
    if filename != st.session_state["uploaded_filename"]:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_filename"] = filename
        st.session_state["original_df"] = df.copy()
        st.session_state["updated_df"] = df.copy()
        st.success(f"✅ File '{filename}' uploaded successfully!")
    else:
        df = st.session_state["original_df"]

    if st.sidebar.button("♻️ Reset to Original"):
        st.session_state["updated_df"] = st.session_state["original_df"].copy()
        st.success("✅ Reset to original file!")
        st.rerun()
else:
    st.info("👈 Upload a CSV file to start.")
    st.stop()

updated_df = st.session_state["updated_df"]

# ----------------- METRICS -----------------
rows, cols = updated_df.shape
missing = updated_df.isna().sum().sum()
c1, c2, c3 = st.columns(3)
c1.metric("Rows", rows)
c2.metric("Columns", cols)
c3.metric("Missing Values", missing)
st.divider()

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🧹 Delete Messy Data",
    "🔧 Data Type Handling",
    "🧮 Missing Value Handling",
    "📏 Scaling",
    "🔠 Encoding",
    "🧠 NLP / Text Cleaning"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab1:
    show_help("tab1", "Data Overview Help", """
    **Purpose:** Review your dataset before cleaning.
    """)

    st.subheader("📊 Dataset Overview")
    st.dataframe(updated_df.head(), use_container_width=True)

    st.write("### Column Information")
    col_info = pd.DataFrame({
        "Column": updated_df.columns,
        "Datatype": updated_df.dtypes.astype(str),
        "Missing Values": updated_df.isnull().sum(),
        "Unique Values": updated_df.nunique(),
    })
    st.dataframe(col_info, use_container_width=True)

    st.write("### Descriptive Statistics")
    st.dataframe(updated_df.describe(include="all").T, use_container_width=True)

# ----------------- TAB 2: DELETE MESSY DATA -----------------
with tab2:
    show_help("tab2", "Delete Messy Data Help", """
    **Purpose:** Remove duplicate or constant columns.
    """)

    st.subheader("🧹 Delete Messy Data")
    cols = list(updated_df.columns)
    selected_cols = st.multiselect("Select Columns to Drop", cols)
    drop_duplicate = st.checkbox("Drop duplicate columns")
    drop_constant = st.checkbox("Drop constant columns")
    threshold = st.slider("Constant threshold", 0.8, 1.0, 0.95)

    if st.button("✅ Apply Tab 2 Changes"):
        df2 = updated_df.copy()
        before_cols = len(df2.columns)

        # Drop selected
        if selected_cols:
            df2.drop(columns=selected_cols, inplace=True)

        # Drop duplicates
        if drop_duplicate:
            df2 = df2.loc[:, ~df2.columns.duplicated(keep="first")]

        # Drop constant
        if drop_constant:
            const_cols = [c for c in df2.columns
                          if df2[c].value_counts(normalize=True, dropna=False).max() >= threshold]
            if const_cols:
                df2.drop(columns=const_cols, inplace=True)
                st.warning(f"Dropped constant columns: {const_cols}")

        st.session_state["updated_df"] = df2
        st.success(f"✅ {before_cols} → {len(df2.columns)} columns after cleanup.")
        st.rerun()

# ----------------- TAB 3: DATA TYPE HANDLING -----------------
with tab3:
    show_help("tab3", "Data Type Handling Help", "Fix incorrect column types.")
    st.subheader("🔧 Data Type Handling")

    st.dataframe(updated_df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Type"}))

    if st.button("⚙️ Auto Detect Data Types"):
        df3 = updated_df.copy()
        for col in df3.columns:
            try:
                if df3[col].dtype == "object":
                    if df3[col].str.replace(".", "", 1).str.isnumeric().all():
                        df3[col] = pd.to_numeric(df3[col], errors="coerce")
                    elif pd.to_datetime(df3[col], errors="coerce").notna().sum() > 0.8 * len(df3[col]):
                        df3[col] = pd.to_datetime(df3[col], errors="coerce")
            except Exception:
                continue
        st.session_state["updated_df"] = df3
        st.success("✅ Auto type detection complete.")
        st.rerun()

    cols = list(updated_df.columns)
    selected_cols = st.multiselect("Select Columns to Convert", cols)
    new_dtype = st.selectbox("Convert To", ["int64", "float64", "object", "category", "datetime64[ns]"])

    if st.button("✅ Apply Conversion"):
        df3 = updated_df.copy()
        for col in selected_cols:
            try:
                if new_dtype == "datetime64[ns]":
                    df3[col] = pd.to_datetime(df3[col], errors="coerce")
                else:
                    df3[col] = df3[col].astype(new_dtype)
            except Exception as e:
                st.warning(f"⚠️ Could not convert {col}: {e}")
        st.session_state["updated_df"] = df3
        st.success("✅ Data type conversion applied.")
        st.rerun()

# ----------------- TAB 4: MISSING VALUE HANDLING -----------------
with tab4:
    show_help("tab4", "Missing Value Handling Help", "Handle missing (null) values.")
    st.subheader("🧮 Missing Value Handling")

    cols = list(updated_df.columns)
    operation = st.selectbox("Operation", ["Drop Null Rows", "Fill Null Values"])
    selected_cols = st.multiselect("Columns", cols)

    if operation == "Fill Null Values":
        strategy = st.selectbox("Strategy", ["mean", "median", "mode", "constant", "ffill", "bfill"])
        const_value = st.text_input("Constant Value (if chosen)")

    if st.button("✅ Apply Missing Value Handling"):
        df4 = updated_df.copy()
        if operation == "Drop Null Rows":
            df4.dropna(inplace=True)
        else:
            for c in (selected_cols if selected_cols else df4.columns):
                try:
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(df4[c]):
                        df4[c].fillna(df4[c].mean(), inplace=True)
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(df4[c]):
                        df4[c].fillna(df4[c].median(), inplace=True)
                    elif strategy == "mode":
                        df4[c].fillna(df4[c].mode()[0], inplace=True)
                    elif strategy == "constant":
                        df4[c].fillna(const_value, inplace=True)
                    elif strategy in ["ffill", "bfill"]:
                        df4[c].fillna(method=strategy, inplace=True)
                except Exception as e:
                    st.warning(f"⚠️ {c}: {e}")
        st.session_state["updated_df"] = df4.reset_index(drop=True)
        st.success("✅ Missing values handled successfully.")
        st.rerun()

# ----------------- TAB 5: SCALING -----------------
with tab5:
    show_help("tab5", "Scaling Help", "Normalize numeric columns.")
    st.subheader("📏 Scaling / Transformation")

    num_cols = updated_df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if not set(updated_df[c].dropna().unique()).issubset({0, 1})]

    if num_cols:
        selected_cols = st.multiselect("Numeric Columns", num_cols)
        method = st.selectbox("Scaling Method",
                              ["None", "StandardScaler", "MinMaxScaler", "RobustScaler", "Log", "Power"])

        if st.button("✅ Apply Scaling"):
            df5 = updated_df.copy()
            for col in selected_cols:
                try:
                    if method == "Log":
                        if (df5[col] <= 0).any():
                            st.warning(f"⚠️ {col} has non-positive values.")
                        else:
                            df5[col] = np.log1p(df5[col])
                    elif method == "StandardScaler":
                        df5[col] = StandardScaler().fit_transform(df5[[col]]).ravel()
                    elif method == "MinMaxScaler":
                        df5[col] = MinMaxScaler().fit_transform(df5[[col]]).ravel()
                    elif method == "RobustScaler":
                        df5[col] = RobustScaler().fit_transform(df5[[col]]).ravel()
                    elif method == "Power":
                        df5[col] = PowerTransformer().fit_transform(df5[[col]]).ravel()
                except Exception as e:
                    st.warning(f"⚠️ Skipped {col}: {e}")
            st.session_state["updated_df"] = df5
            st.success("✅ Scaling applied.")
            st.rerun()
    else:
        st.info("ℹ️ No numeric columns available.")

# ----------------- TAB 6: ENCODING -----------------
with tab6:
    show_help("tab6", "Encoding Help", "Convert categorical columns to numeric.")
    st.subheader("🔠 Encoding Data")

    cat_cols = updated_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        selected_cols = st.multiselect("Categorical Columns", cat_cols)
        method = st.selectbox("Encoding Method", ["None", "Label", "Ordinal", "OneHot"])
        if st.button("✅ Apply Encoding"):
            df6 = updated_df.copy()
            try:
                if selected_cols and method != "None":
                    if method == "Label":
                        for col in selected_cols:
                            df6[col] = LabelEncoder().fit_transform(df6[col].astype(str))
                    elif method == "Ordinal":
                        df6[selected_cols] = OrdinalEncoder().fit_transform(df6[selected_cols].astype(str))
                    elif method == "OneHot":
                        ohe = OneHotEncoder(sparse_output=False)
                        encoded = pd.DataFrame(
                            ohe.fit_transform(df6[selected_cols]),
                            columns=ohe.get_feature_names_out(selected_cols),
                            index=df6.index
                        )
                        df6 = pd.concat([df6.drop(columns=selected_cols), encoded], axis=1)
            except Exception as e:
                st.warning(f"⚠️ Encoding failed: {e}")
            st.session_state["updated_df"] = df6
            st.success("✅ Encoding complete.")
            st.rerun()
    else:
        st.info("ℹ️ No categorical columns found.")

# ----------------- TAB 7: NLP / TEXT CLEANING -----------------
with tab7:
    show_help("tab7", "NLP / Text Cleaning Help", """
    **Purpose:** Clean and preprocess text columns for NLP tasks.

    **Features:**
    - **Lowercase:** Convert all text to lowercase
    - **Remove Punctuation:** Strip punctuation marks
    - **Remove Numbers:** Remove all numeric characters
    - **Remove Stopwords:** Remove common words (the, is, at, etc.)
    - **Remove Extra Whitespace:** Clean multiple spaces
    - **Remove URLs:** Remove web links
    - **Remove Email:** Remove email addresses
    - **Lemmatization:** Convert words to base form (running → run)
    - **Stemming:** Reduce words to root form (running → run)
    - **Remove Special Characters:** Keep only alphanumeric and spaces
    """)

    st.subheader(" NLP / Text Cleaning")

    text_cols = updated_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if text_cols:
        selected_cols = st.multiselect("Select Text Columns to Clean", text_cols, key="nlp_cols")

        st.write("#### Select Cleaning Operations:")
        col1, col2 = st.columns(2)

        with col1:
            lowercase = st.checkbox("Convert to lowercase", value=True)
            remove_punct = st.checkbox("Remove punctuation", value=True)
            remove_numbers = st.checkbox("Remove numbers")
            remove_whitespace = st.checkbox("Remove extra whitespace", value=True)
            remove_urls = st.checkbox("Remove URLs")

        with col2:
            remove_emails = st.checkbox("Remove email addresses")
            remove_special = st.checkbox("Remove special characters")
            remove_stop = st.checkbox("Remove stopwords")
            lemmatize = st.checkbox("Apply lemmatization")
            stemming = st.checkbox("Apply stemming")

        if st.button("✅ Apply NLP Cleaning"):
            if not selected_cols:
                st.warning("⚠️ Please select at least one text column to clean.")
            else:
                df7 = updated_df.copy()
                stop_words = set(stopwords.words("english"))
                lemmatizer = WordNetLemmatizer()
                stemmer = PorterStemmer()

                def clean_text(text):
                    if not isinstance(text, str):
                        return text

                    # Remove URLs
                    if remove_urls:
                        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

                    # Remove emails
                    if remove_emails:
                        text = re.sub(r'\S+@\S+', '', text)

                    # Convert to lowercase
                    if lowercase:
                        text = text.lower()

                    # Remove punctuation
                    if remove_punct:
                        text = text.translate(str.maketrans("", "", string.punctuation))

                    # Remove numbers
                    if remove_numbers:
                        text = re.sub(r"\d+", "", text)

                    # Remove special characters
                    if remove_special:
                        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

                    # Remove extra whitespace
                    if remove_whitespace:
                        text = re.sub(r'\s+', ' ', text).strip()

                    # Remove stopwords
                    if remove_stop:
                        text = " ".join([w for w in text.split() if w not in stop_words])

                    # Lemmatization
                    if lemmatize:
                        text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])

                    # Stemming
                    if stemming:
                        text = " ".join([stemmer.stem(w) for w in text.split()])

                    return text.strip()

                for col in selected_cols:
                    try:
                        df7[col] = df7[col].astype(str).apply(clean_text)
                    except Exception as e:
                        st.warning(f"⚠️ Error cleaning {col}: {e}")

                st.session_state["updated_df"] = df7
                st.success("✅ NLP text cleaning applied successfully.")
                st.rerun()
    else:
        st.info("ℹ️ No text columns available for NLP cleaning.")

# ----------------- FINAL PREVIEW & DOWNLOAD -----------------
st.divider()
st.header("📊 Final Data Preview")
st.dataframe(st.session_state["updated_df"].head(), use_container_width=True)

if mode == "Download":
    csv = st.session_state["updated_df"].to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
