conda activate survival-env

python
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings
import streamlit.components.v1 as components  # 添加缺少的导入

# 加载模型
model = joblib.load('RF.pkl')
X_test = pd.read_csv('X_test.csv')

# 设置英文字体
plt.rcParams['font.family'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
warnings.filterwarnings("ignore")

# 使用已存在的X_test数据
# 假设X_test已经在全局命名空间中定义
# 如果X_test还未定义，请取消下面的注释并根据实际情况调整
# X_test = test_processed.drop(['OS', 'OS.time'], axis=1)

# 从X_test获取特征名称
feature_names = X_test.columns.tolist()

# 设置网页标题
st.title("Medical Risk Predictor")

# 创建输入字段，根据特征名称和合理范围设置
PT = st.number_input("PT Value:", min_value=0.0, max_value=3000.0, value=12.0, step=0.1)
APTT = st.number_input("APTT Value:", min_value=0.0, max_value=3000.0, value=28.0, step=0.1)
MCHC = st.number_input("MCHC Value:", min_value=0.0, max_value=3000.0, value=310.0, step=1.0)
Lymphpct = st.number_input("Lymphocyte Percentage:", min_value=0.0, max_value=3000.0, value=12.0, step=0.01)
Monopct = st.number_input("Monocyte Percentage:", min_value=0.0, max_value=3000.0, value=3.0, step=0.01)
Neutpct = st.number_input("Neutrophil Percentage:", min_value=0.0, max_value=3000.0, value=83.0, step=0.01)
Neut = st.number_input("Neutrophil Count:", min_value=0.0, max_value=3000.0, value=4.0, step=0.01)
Eospct = st.number_input("Eosinophil Percentage:", min_value=0.0, max_value=3000.0, value=0.3, step=0.01)
Basopct = st.number_input("Basophil Percentage:", min_value=0.0, max_value=3000.0, value=0.05, step=0.01)
Hb = st.number_input("Hemoglobin:", min_value=0.0, max_value=3000.0, value=11.0, step=0.01)
PLT = st.number_input("Platelet Count:", min_value=0.0, max_value=3000.0, value=170.0, step=1.0)
TBIL = st.number_input("Total Bilirubin:", min_value=0.0, max_value=3000.0, value=1.0, step=0.01)
ALB = st.number_input("Albumin:", min_value=0.0, max_value=3000.0, value=2.9, step=0.01)
Cr = st.number_input("Creatinine:", min_value=0.0, max_value=3000.0, value=5.0, step=0.01)
Ur = st.number_input("Uric Acid:", min_value=0.0, max_value=3000.0, value=0.2, step=0.01)
K = st.number_input("Potassium:", min_value=0.0, max_value=3000.0, value=5.2, step=0.01)
Na = st.number_input("Sodium:", min_value=0.0, max_value=3000.0, value=135.0, step=0.1)
Ca = st.number_input("Calcium:", min_value=0.0, max_value=3000.0, value=2.0, step=0.01)
P = st.number_input("Phosphorus:", min_value=0.0, max_value=3000.0, value=1.2, step=0.01)
Glu = st.number_input("Glucose:", min_value=0.0, max_value=3000.0, value=1380.0, step=1.0)
LDH = st.number_input("Lactate Dehydrogenase:", min_value=0.0, max_value=3000.0, value=950.0, step=1.0)
AG = st.number_input("Anion Gap:", min_value=0.0, max_value=3000.0, value=16.0, step=0.1)
WBC = st.number_input("White Blood Cell Count:", min_value=0.0, max_value=3000.0, value=5.0, step=0.1)

# 将用户输入的特征值存入列表
feature_values = [
    PT, APTT, MCHC, Lymphpct, Monopct, Neutpct, Neut, Eospct, Basopct, Hb, PLT, 
    TBIL, ALB, Cr, Ur, K, Na, Ca, P, Glu, LDH, AG, WBC
]

# 将特征转换为NumPy数组，适用于模型输入
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别
    predicted_class = model.predict(features)[0]
    
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]
    
    st.write(f"**Prediction Result:** {'High Risk' if predicted_class == 1 else 'Low Risk'} (1: High Risk, 0: Low Risk)")
    st.write(f"**Prediction Probability:** {predicted_proba}")
    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:
        advice = (
            f"According to the model prediction, you have a higher risk of the condition."
            f"The model predicts your risk probability to be {probability:.1f}%. "
            "It is recommended that you consult a medical professional for further evaluation and possible interventions."
        )
    else:
        advice = (
            f"According to the model prediction, your risk of the condition is low."
            f"The model predicts a {probability:.1f}% probability of no risk. "
            "However, maintaining a healthy lifestyle is important. Please continue to have regular check-ups with your healthcare provider."
        )
    
    st.write(advice)
    
    # SHAP解释
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], 
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], 
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
    
    # LIME解释
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Low Risk', 'High Risk'],
        mode='classification'
    )
    
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )
    
    lime_html = lime_exp.as_html(show_table=False)
    components.html(lime_html, height=800, scrolling=True)  # 修改为使用导入的components