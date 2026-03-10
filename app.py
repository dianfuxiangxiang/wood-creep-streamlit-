import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="Wood Creep Curve Visualization",
    page_icon="📈",
    layout="wide"
)

# =========================
# 模型结构（必须和训练时一致）
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, in_dim=5, hidden=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


# =========================
# 常量
# =========================
MODEL_PATH = "model.pt"
STATS_PATH = "stats.npz"

T_MIN, T_MAX, DT = 0, 45, 1
GRID_T = np.arange(T_MIN, T_MAX + 1, DT).astype(float)  # 46 points


# =========================
# 加载模型与标准化参数
# =========================
@st.cache_resource
def load_model_and_stats():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Cannot find {MODEL_PATH}")
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"Cannot find {STATS_PATH}")

    stats = np.load(STATS_PATH)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]

    model = LSTMRegressor(in_dim=5, hidden=128, num_layers=2, dropout=0.1)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, x_mean, x_std


def build_input_sequence(rc, rd, mc, temp):
    """
    构建输入序列 X: (46, 5)
    [time, Rc, Rd, MC, Temp]
    """
    X = np.stack(
        [
            GRID_T,
            np.full_like(GRID_T, rc, dtype=float),
            np.full_like(GRID_T, rd, dtype=float),
            np.full_like(GRID_T, mc, dtype=float),
            np.full_like(GRID_T, temp, dtype=float),
        ],
        axis=1
    )
    return X


def normalize_x(X, x_mean, x_std):
    return (X - x_mean) / x_std


def predict_normalized_curve(model, X_norm):
    """
    输入:
        X_norm: (46, 5)
    输出:
        y_norm_pred: (46,)
    """
    x_tensor = torch.tensor(X_norm[None, :, :], dtype=torch.float32)  # (1,46,5)
    with torch.no_grad():
        y_pred = model(x_tensor).cpu().numpy()[0, :, 0]
    return y_pred


def reconstruct_absolute_curve(y_norm_pred, y0, scale):
    """
    用用户输入的 y0 和 scale 恢复绝对曲线
    """
    return y0 + y_norm_pred * scale


def make_curve_dataframe(time_arr, curve_arr, curve_name):
    return pd.DataFrame({
        "time_min": time_arr,
        curve_name: curve_arr
    })


# =========================
# 标题
# =========================
st.title("Wood Creep Curve Visualization")
st.markdown(
    """
This app visualizes the **predicted creep/strain curve** based on the trained **LSTM-PINN style sequence model**.

**Current deployment note**  
The saved model predicts a **normalized relative curve**.  
If you also provide **initial strain (y0)** and **curve scale**, the app can reconstruct an absolute curve.
"""
)

# =========================
# 加载模型
# =========================
try:
    model, x_mean, x_std = load_model_and_stats()
    st.success("Model and normalization stats loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model or stats: {e}")
    st.stop()

# =========================
# 侧边栏输入
# =========================
st.sidebar.header("Input Parameters")

rc = st.sidebar.number_input("Rc", value=0.50, format="%.6f")
rd = st.sidebar.number_input("Rd", value=0.50, format="%.6f")
mc = st.sidebar.number_input("MC", value=0.10, format="%.6f")
temp = st.sidebar.number_input("Temperature", value=25.0, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.subheader("Optional absolute reconstruction")

use_absolute = st.sidebar.checkbox("Reconstruct absolute curve", value=False)

y0 = None
scale = None

if use_absolute:
    y0 = st.sidebar.number_input("Initial strain (y0)", value=0.010000, format="%.6f")
    scale = st.sidebar.number_input("Curve scale", value=0.020000, min_value=0.000001, format="%.6f")

# =========================
# 主按钮
# =========================
if st.button("Predict Curve"):
    # 1. 构建输入
    X = build_input_sequence(rc, rd, mc, temp)
    X_norm = normalize_x(X, x_mean, x_std)

    # 2. 预测归一化曲线
    y_norm_pred = predict_normalized_curve(model, X_norm)

    # 3. 输出结果
    col1, col2 = st.columns([1.2, 1.0])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))

        if use_absolute:
            y_abs = reconstruct_absolute_curve(y_norm_pred, y0, scale)
            ax.plot(GRID_T, y_abs, linewidth=2.2, label="Predicted absolute curve")
            ax.set_ylabel("Predicted strain")
            curve_df = make_curve_dataframe(GRID_T, y_abs, "predicted_strain")
        else:
            ax.plot(GRID_T, y_norm_pred, linewidth=2.2, label="Predicted normalized curve")
            ax.set_ylabel("Normalized relative strain")
            curve_df = make_curve_dataframe(GRID_T, y_norm_pred, "predicted_normalized_curve")

        ax.set_xlabel("Time (min)")
        ax.set_title("Predicted Curve")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Input Summary")
        st.write(f"**Rc** = {rc:.6f}")
        st.write(f"**Rd** = {rd:.6f}")
        st.write(f"**MC** = {mc:.6f}")
        st.write(f"**Temperature** = {temp:.6f}")

        st.subheader("Key Output Values")
        if use_absolute:
            st.write(f"**t = 1 min**: {y_abs[1]:.6f}")
            st.write(f"**t = 45 min**: {y_abs[-1]:.6f}")
            st.write(f"**Maximum predicted strain**: {np.max(y_abs):.6f}")
            st.write(f"**Minimum predicted strain**: {np.min(y_abs):.6f}")
        else:
            st.write(f"**t = 1 min (normalized)**: {y_norm_pred[1]:.6f}")
            st.write(f"**t = 45 min (normalized)**: {y_norm_pred[-1]:.6f}")
            st.write(f"**Maximum normalized value**: {np.max(y_norm_pred):.6f}")
            st.write(f"**Minimum normalized value**: {np.min(y_norm_pred):.6f}")

    st.subheader("Prediction Data")
    st.dataframe(curve_df, use_container_width=True)

    csv_data = curve_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download prediction as CSV",
        data=csv_data,
        file_name="predicted_curve.csv",
        mime="text/csv"
    )

# =========================
# 底部说明
# =========================
st.markdown("---")
st.markdown(
    """
### Deployment file checklist
Make sure the repository contains:

- `app.py`
- `requirements.txt`
- `model.pt`
- `stats.npz`

### Important note
The current saved model predicts a **normalized relative sequence**.  
For a fully physical/absolute curve prediction from only `Rc/Rd/MC/T`, you would typically also need a model for:
- initial strain `y0`
- curve amplitude/scale
"""
)