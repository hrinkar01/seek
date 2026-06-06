import streamlit as st
import pandas as pd

from seek import auto_train_model

# Page config
st.set_page_config(
    page_title="Seek",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 Seek")
st.markdown(
    "Upload a dataset, select a target column, and let Seek automatically "
    "clean the data, preprocess features, train multiple models, and compare results."
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # Dataset info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    # Select target
    target_column = st.selectbox(
        "Select Target Column",
        options=df.columns
    )

    # Train button
    if st.button("🚀 Train Models"):

        with st.spinner("Training models..."):

            try:
                results_df, best_model, best_score = auto_train_model(
                    df,
                    target_column
                )

                st.success("Training completed!")

                st.subheader("🏆 Best Model")

                st.metric(
                    label=best_model,
                    value=f"{best_score:.2%}"
                )

                st.subheader("📊 Model Leaderboard")

                st.dataframe(
                    results_df,
                    width='stretch'
                )

            except Exception as e:
                st.error(f"Error: {e}")