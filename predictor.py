import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 加载模型
model = joblib.load('RF.pkl')
train_processed = train.drop(columns=variables_to_drop, errors="ignore")
test_processed = test.drop(columns=variables_to_drop, errors="ignore")

X_test = test_processed.drop(['OS', 'OS.time'], axis=1)


# 使用已存在的X_test数据
# 假设X_test已经在全局命名空间中定义
# 如果X_test还未定义，请取消下面的注释并根据实际情况调整
# X_test = test_processed.drop(['OS', 'OS.time'], axis=1)

# 从X_test获取特征名称
feature_names = X_test.columns.tolist()

# 设置网页标题
st.title("医学风险预测器")

# 创建输入字段，根据特征名称和合理范围设置
PT = st.number_input("PT值:", min_value=0.0, max_value=3000.0, value=12.0, step=0.1)
APTT = st.number_input("APTT值:", min_value=0.0, max_value=3000.0, value=28.0, step=0.1)
MCHC = st.number_input("MCHC值:", min_value=0.0, max_value=3000.0, value=310.0, step=1.0)
Lymphpct = st.number_input("淋巴细胞百分比:", min_value=0.0, max_value=3000.0, value=12.0, step=0.01)
Monopct = st.number_input("单核细胞百分比:", min_value=0.0, max_value=3000.0, value=3.0, step=0.01)
Neutpct = st.number_input("中性粒细胞百分比:", min_value=0.0, max_value=3000.0, value=83.0, step=0.01)
Neut = st.number_input("中性粒细胞计数:", min_value=0.0, max_value=3000.0, value=4.0, step=0.01)
Eospct = st.number_input("嗜酸性粒细胞百分比:", min_value=0.0, max_value=3000.0, value=0.3, step=0.01)
Basopct = st.number_input("嗜碱性粒细胞百分比:", min_value=0.0, max_value=3000.0, value=0.05, step=0.01)
Hb = st.number_input("血红蛋白:", min_value=0.0, max_value=3000.0, value=11.0, step=0.01)
PLT = st.number_input("血小板计数:", min_value=0.0, max_value=3000.0, value=170.0, step=1.0)
TBIL = st.number_input("总胆红素:", min_value=0.0, max_value=3000.0, value=1.0, step=0.01)
ALB = st.number_input("白蛋白:", min_value=0.0, max_value=3000.0, value=2.9, step=0.01)
Cr = st.number_input("肌酐:", min_value=0.0, max_value=3000.0, value=5.0, step=0.01)
Ur = st.number_input("尿酸:", min_value=0.0, max_value=3000.0, value=0.2, step=0.01)
K = st.number_input("钾:", min_value=0.0, max_value=3000.0, value=5.2, step=0.01)
Na = st.number_input("钠:", min_value=0.0, max_value=3000.0, value=135.0, step=0.1)
Ca = st.number_input("钙:", min_value=0.0, max_value=3000.0, value=2.0, step=0.01)
P = st.number_input("磷:", min_value=0.0, max_value=3000.0, value=1.2, step=0.01)
Glu = st.number_input("葡萄糖:", min_value=0.0, max_value=3000.0, value=1380.0, step=1.0)
LDH = st.number_input("乳酸脱氢酶:", min_value=0.0, max_value=3000.0, value=950.0, step=1.0)
AG = st.number_input("阴离子间隙:", min_value=0.0, max_value=3000.0, value=16.0, step=0.1)
WBC = st.number_input("白细胞计数:", min_value=0.0, max_value=3000.0, value=5.0, step=0.1)

# 将用户输入的特征值存入列表
feature_values = [
    PT, APTT, MCHC, Lymphpct, Monopct, Neutpct, Neut, Eospct, Basopct, Hb, PLT, 
    TBIL, ALB, Cr, Ur, K, Na, Ca, P, Glu, LDH, AG, WBC
]

# 将特征转换为NumPy数组，适用于模型输入
features = np.array([feature_values])

if st.button("预测"):
    # 预测类别
    predicted_class = model.predict(features)[0]
    
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]
    
    st.write(f"**预测结果:** {predicted_class} (1: 风险, 0: 低风险)")
    st.write(f"**预测概率:** {predicted_proba}")
    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:
        advice = (
            f"根据模型预测，您有较高的风险。"
            f"模型预测您的风险概率为 {probability:.1f}%。 "
            "建议您咨询医疗专业人士进行进一步评估和可能的干预措施。"
        )
    else:
        advice = (
            f"根据模型预测，您的风险较低。"
            f"模型预测您无风险的概率为 {probability:.1f}%。 "
            "不过，保持健康的生活方式很重要。请继续定期与您的医疗服务提供者进行检查。"
        )
    
    st.write(advice)
    
    # SHAP解释
    st.subheader("SHAP Force Plot 解释")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], 
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], 
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot 解释')
    
    # LIME解释
    st.subheader("LIME 解释")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['低风险', '高风险'],
        mode='classification'
    )
    
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )
    
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)