# file: app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer,
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="🧮 VizAi Data Cleaning", layout="wide")
st.title("🧮 VizAi Data Cleaning- Ethicallogix")

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🧹 Delete Messy Data",
    "🔧 Data Type Handling",
    "🧮 Missing Value Handling",
    "📏 Scaling",
    "🔠 Encoding"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab1:
    show_help("tab1", "Data Overview Help", """
    **Purpose:**  
    This tab helps you understand your dataset before cleaning or processing.

    **Features Explained:**  
    - 🧾 **Dataset Preview:** Displays the first few rows to quickly inspect your data.  
    - 📊 **Column Information:** Shows each column name, its data type, number of missing values, and count of unique values.  
    - 📈 **Descriptive Statistics:** Summarizes numeric columns (mean, std, min, max, etc.).  

    **How to Use:**  
    1. Review the preview to understand dataset structure.  
    2. Check column types to spot incorrect types (like numbers stored as text).  
    3. Examine descriptive stats to detect outliers or inconsistent data.
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
    **Purpose:**  
    Remove unnecessary, duplicate, or constant columns to clean your dataset.

    **Functions Used:**  
    - `df.drop(columns=...)`: Removes selected columns.  
    - `df.columns.duplicated()`: Detects duplicate column names.  
    - `value_counts(normalize=True)`: Finds columns with constant values.  

    **How to Use:**  
    1. Select columns manually to delete.  
    2. Use checkboxes to remove duplicate or constant columns.  
    3. Adjust the threshold (e.g., 0.95 means 95%+ identical values will be dropped).  
    4. Click **Apply** to clean your dataset.
    """)
    st.subheader("🧹 Delete Messy Data")

    cols = list(updated_df.columns)
    selected_cols = st.multiselect("Select Columns to Drop", cols)
    drop_selected = st.checkbox("Drop selected columns")
    drop_duplicate = st.checkbox("Drop duplicate columns")
    drop_constant = st.checkbox("Drop constant columns")
    threshold = st.slider("Constant threshold", 0.8, 1.0, 0.95)

    if st.button("✅ Apply Tab 2 Changes"):
        df2 = updated_df.copy()
        before_cols = len(df2.columns)
        if drop_selected and selected_cols:
            df2.drop(columns=selected_cols, inplace=True)
        if drop_duplicate:
            df2 = df2.loc[:, ~df2.columns.duplicated(keep="first")]
        if drop_constant:
            const_cols = [c for c in df2.columns if
                          df2[c].value_counts(normalize=True, dropna=False).max() >= threshold]
            if const_cols:
                df2.drop(columns=const_cols, inplace=True)
                st.warning(f"Dropped constant columns: {const_cols}")
        st.session_state["updated_df"] = df2
        st.success(f"✅ {before_cols} → {len(df2.columns)} columns after cleanup.")
        st.rerun()

# ----------------- TAB 3: DATA TYPE HANDLING -----------------
with tab3:
    show_help("tab3", "Data Type Handling Help", """
    **Purpose:**  
    Fix incorrect column data types (e.g., text stored as numbers or dates).

    **Functions Used:**  
    - `pd.to_numeric()`: Converts strings to numeric type.  
    - `pd.to_datetime()`: Converts date-like text to datetime type.  
    - `df.astype()`: Manually change data type.  

    **How to Use:**  
    1. Click **Auto Detect** to automatically convert columns (like numbers or dates stored as text).  
    2. Use dropdowns to manually select columns and target data type.  
    3. Apply changes to update your dataset.
    """)
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
    show_help("tab4", "Missing Value Handling Help", """
    **Purpose:**  
    Handle missing (null) values to make data usable for analysis.

    **Functions Used:**  
    - `df.dropna()`: Removes rows with null values.  
    - `df.fillna(value)`: Fills missing values with specified value.  
    - `df.fillna(method='ffill'/'bfill')`: Propagates values forward or backward.  

    **Strategies:**  
    - **Mean/Median:** Replace missing numeric values with column average.  
    - **Mode:** Fill with most frequent value.  
    - **Constant:** Replace with a fixed custom value.  
    - **ffill/bfill:** Fill missing values using previous/next row values.  

    **How to Use:**  
    1. Choose columns and a strategy.  
    2. Apply operation to clean missing data.
    """)
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
    show_help("tab5", "Scaling Help", """
    **Purpose:**  
    Scale numerical columns to normalize their range and reduce bias in models.

    **Functions Used:**  
    - `StandardScaler()`: Mean = 0, Std = 1 normalization.  
    - `MinMaxScaler()`: Scales between 0 and 1.  
    - `RobustScaler()`: Resistant to outliers.  
    - `np.log1p()`: Logarithmic transformation.  
    - `PowerTransformer()`: Gaussian-like transformation.  

    **How to Use:**  
    1. Select numeric columns.  
    2. Choose a scaling method.  
    3. Apply and verify normalized data.
    """)
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
    show_help("tab6", "Encoding Help", """
    **Purpose:**  
    Convert categorical (text) columns into numeric format for ML models.

    **Functions Used:**  
    - `LabelEncoder()`: Converts categories into 0...N integers.  
    - `OrdinalEncoder()`: Encodes multiple columns ordinally.  
    - `OneHotEncoder()`: Creates binary columns for each category.  

    **How to Use:**  
    1. Select text columns.  
    2. Choose encoding method:  
       - **Label:** For single-label categorical columns.  
       - **Ordinal:** For ordered categorical data.  
       - **OneHot:** For non-ordered categories.  
    3. Apply and verify encoded columns.
    """)
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

# ----------------- PREVIEW OR DOWNLOAD -----------------
st.divider()
st.header("📊 Final Data Preview")
st.dataframe(st.session_state["updated_df"].head(), use_container_width=True)

if mode == "Download":
    csv = st.session_state["updated_df"].to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
