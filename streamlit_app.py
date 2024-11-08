import pandas as pd
import streamlit as st # type: ignore

st.markdown("# データ分析アプリ")

uploaded_file  = st.file_uploader("ファイルを入力してください")
if uploaded_file is not None:
    st.info("ファイルが正しくアップロードされました")

    df = pd.read_csv(uploaded_file)
    st.table(df.head(5))
    
    column_names = [c for c in df.columns]
    target_column = st.selectbox(
        'ターゲットを選んでください',
        column_names
    )
    st.info(f"{target_column}を予測対象として予測モデルを作ります！")

