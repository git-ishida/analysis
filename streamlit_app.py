import pandas as pd
import streamlit as st # type: ignore
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


if 'target_column' not in st.session_state:
    st.session_state['target_column'] = 'ターゲット指定なし'


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

    # 'モデル作成開始'ボタンがクリックされるか、ターゲットカラムが変更されない場合にifを実行
    if st.button('モデル作成開始') or st.session_state['target_column'] == target_column:
        st.session_state['target_column'] = target_column
        x = df.copy()
        y = x[target_column]
        x = x.drop([target_column], axis=1)

        model = DecisionTreeRegressor(random_state=777, max_depth=3)
        model.fit(x, y)
        st.info("予測モデル作成が完了しました！")


        st.markdown("### 散布図の確認")
        col_x = st.selectbox('x軸にする列を選んでください', column_names)
        col_y = st.selectbox('y軸にする列を選んでください', column_names, index=1)

        fig = plt.figure(figsize=(5, 5))
        sns.scatterplot(data=df, x=col_x, y=col_y)
        st.pyplot(fig)

