import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import traceback

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('RF.pkl')

model = load_model()

# 获取模型特征名称
if hasattr(model, 'feature_names_in_'):
    MODEL_FEATURES = list(model.feature_names_in_)
    st.sidebar.write(f"模型期望 {len(MODEL_FEATURES)} 个特征")
    st.sidebar.write(MODEL_FEATURES)
else:
    # 如果没有特征名称，使用原始定义
    MODEL_FEATURES = ["Age", "AAPR", "LAR", "Hemoglobin", "GCS", "SOFA", "PLR", "PNI"]

# 特征范围定义 - 根据模型特征重新组织
feature_ranges = {
    "Age": {"type": "numerical", "min": 18, "max": 89, "default": 45},
    "AAPR": {"type": "numerical", "min": 0.0341, "max": 1.2875, "default": 0.5000},
    "LAR": {"type": "numerical", "min": 0.0132, "max": 0.6726, "default": 0.3000},
    "Hemoglobin": {"type": "numerical", "min": 28.5, "max": 195, "default": 100},
    "GCS": {"type": "numerical", "min": 1, "max": 15, "default": 12},
    "SOFA": {"type": "numerical", "min": 2, "max": 24, "default": 5},
    "PLR": {"type": "numerical", "min": 10.674, "max": 950.000, "default": 400.00},
    "PNI": {"type": "numerical", "min": 16.55, "max": 442.45, "default": 200.00},
}

# Streamlit 界面
st.title("Cognitive Impairment Prediction Model with SHAP Visualization")

# 动态生成输入项 - 按照模型特征顺序
st.header("Enter the following feature values:")
user_inputs = {}

for feature in MODEL_FEATURES:
    if feature in feature_ranges:
        props = feature_ranges[feature]
        if props["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({props['min']} - {props['max']})",
                min_value=float(props["min"]),
                max_value=float(props["max"]),
                value=float(props["default"]),
                key=f"input_{feature}"  # 添加唯一的key
            )
        user_inputs[feature] = value
    else:
        st.warning(f"特征 '{feature}' 没有定义范围，请输入值：")
        value = st.number_input(
            label=f"{feature}",
            value=0.0,
            key=f"input_{feature}"
        )
        user_inputs[feature] = value

# 转换为模型输入格式 - 确保正确顺序
feature_values = [user_inputs[feat] for feat in MODEL_FEATURES]
feature_df = pd.DataFrame([feature_values], columns=MODEL_FEATURES)

st.write("### 输入的特征数据预览：")
st.dataframe(feature_df)

# 预测与 SHAP 可视化
if st.button("Predict"):
    try:
        # 模型预测
        predicted_class = model.predict(feature_df)[0]
        predicted_proba = model.predict_proba(feature_df)[0]

        # 提取预测的类别概率
        probability = predicted_proba[predicted_class] * 100

        # 显示预测结果
        st.success(f"### 预测结果")
        st.write(f"**认知障碍预测概率：{probability:.2f}%**")
        st.write(f"预测类别：{predicted_class}")

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        
        # 处理 SHAP 可视化
        st.write("### SHAP 解释")
        
        try:
            if isinstance(shap_values, list):
                # 多分类情况
                shap_array = shap_values[predicted_class]
                expected_value = explainer.expected_value[predicted_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                # 二分类或回归情况
                shap_array = shap_values
                expected_value = explainer.expected_value
            
            # 生成 SHAP 力图
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.force_plot(
                expected_value,
                shap_array[0],
                feature_df.iloc[0],
                matplotlib=True,
                show=False,
                ax=ax
            )
            st.pyplot(fig)
            plt.close()
            
        except Exception as shap_error:
            st.warning(f"SHAP 图生成失败：{str(shap_error)}")
            # 显示特征重要性作为备选
            if hasattr(model, 'feature_importances_'):
                st.write("### 特征重要性")
                importance_df = pd.DataFrame({
                    'Feature': MODEL_FEATURES,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                # 添加数值标签
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', ha='left', va='center')
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"❌ 预测过程中发生错误：")
        st.code(traceback.format_exc())
        
        # 尝试使用 numpy 数组进行预测
        st.write("### 尝试使用备选方法进行预测...")
        try:
            feature_array = feature_df.values.astype(float)
            predicted_class = model.predict(feature_array)[0]
            predicted_proba = model.predict_proba(feature_array)[0]
            probability = predicted_proba[predicted_class] * 100
            st.success(f"备选方法预测成功！概率：{probability:.2f}%")
        except:
            st.error("备选方法也失败了，请检查模型和输入数据。")