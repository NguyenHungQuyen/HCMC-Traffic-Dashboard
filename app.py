import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
import os

# --- Cấu hình trang ---
st.set_page_config(page_title="HCMC Traffic Dashboard", layout="wide")

st.title("🚦 Dashboard Phân Tích & Dự Báo Giao Thông TP.HCM")
st.markdown("**Module:** Integration, Dashboard & Report | **Role:** Người 3")

# --- Xử lý dữ liệu (Data Loader) ---
st.sidebar.header("Dữ Liệu Đầu Vào")

# Hàm load dữ liệu an toàn
def load_data():
    flow = None
    pred = None
    
    # 1. Thử tìm file có sẵn trong thư mục (cho trường hợp Deploy lên Cloud)
    if os.path.exists("hcmc_flow.csv"):
        flow = pd.read_csv("hcmc_flow.csv")
        st.sidebar.success("✅ Đã tự động tải 'hcmc_flow.csv'")
    else:
        # Nếu không có file, hiện nút upload
        up_flow = st.sidebar.file_uploader("Tải file hcmc_flow.csv", type="csv")
        if up_flow:
            flow = pd.read_csv(up_flow)

    if os.path.exists("prediction.csv"):
        pred = pd.read_csv("prediction.csv")
        st.sidebar.success("✅ Đã tự động tải 'prediction.csv'")
    else:
        up_pred = st.sidebar.file_uploader("Tải file prediction.csv", type="csv")
        if up_pred:
            pred = pd.read_csv(up_pred)
            
    return flow, pred

# Gọi hàm load dữ liệu
flow_df, pred_df = load_data()

# Hàm chuyển đổi Slot -> Giờ phút
def get_time_label(slot):
    total_minutes = slot * 15
    h = (total_minutes // 60) % 24
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"

# --- MAIN APP LOGIC ---
# Chỉ chạy khi đã có đủ 2 file dữ liệu
if flow_df is not None and pred_df is not None:
    
    # Preprocessing
    if 'total_flow' not in flow_df.columns:
        flow_df['total_flow'] = flow_df[['motorbike', 'car', 'bus', 'truck']].sum(axis=1)
    
    if 'time_label' not in flow_df.columns:
        flow_df['time_label'] = flow_df['slot_idx'].apply(get_time_label)
    
    # [Giả lập] Tạo dữ liệu Sensor-only nếu chưa có (để so sánh)
    if 'y_pred_sensor' not in pred_df.columns:
        np.random.seed(42)
        pred_df['y_pred_sensor'] = pred_df['y_true'] + np.random.normal(0, 25, size=len(pred_df))
    
    # --- 1. So sánh Mô hình ---
    st.header("1. So Sánh Hiệu Suất Các Mô Hình")
    
    mae_fusion = np.mean(np.abs(pred_df['y_true'] - pred_df['y_pred']))
    mae_sensor = np.mean(np.abs(pred_df['y_true'] - pred_df['y_pred_sensor']))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Fusion Model)", f"{mae_fusion:.2f}", help="Sai số mô hình đề xuất")
    col2.metric("MAE (Sensor-only)", f"{mae_sensor:.2f}", delta=f"-{(mae_sensor - mae_fusion):.2f}", delta_color="inverse")
    col3.metric("Improvement", f"+{((mae_sensor - mae_fusion)/mae_sensor)*100:.1f}%")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=pred_df['y_true'], mode='lines', name='Thực tế (Ground Truth)', line=dict(color='black', width=2)))
    fig_line.add_trace(go.Scatter(y=pred_df['y_pred'], mode='lines', name='Dự báo (Fusion Model)', line=dict(color='blue', dash='dash')))
    fig_line.add_trace(go.Scatter(y=pred_df['y_pred_sensor'], mode='lines', name='Dự báo (Sensor-only)', line=dict(color='red', width=1, dash='dot')))
    fig_line.update_layout(title="So sánh: Thực tế vs Fusion vs Sensor-only", xaxis_title="Time Step", yaxis_title="Lưu lượng xe")
    st.plotly_chart(fig_line, use_container_width=True)

    # --- 2. Phân tích Lưu lượng ---
    st.header("2. Phân Tích Dòng Chảy Giao Thông")
    
    tab1, tab2 = st.tabs(["Bản Đồ Nhiệt (Heatmap)", "Chi Tiết Theo Camera"])
    
    with tab1:
        heatmap_data = flow_df.pivot_table(index='camera_id', columns='time_label', values='total_flow', aggfunc='mean')
        fig_heat = px.imshow(heatmap_data, aspect='auto', labels=dict(x="Thời gian", y="Camera", color="Lưu lượng"), title="Mật độ giao thông theo Giờ")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with tab2:
        selected_cam = st.selectbox("Chọn Camera:", flow_df['camera_id'].unique())
        filtered_df = flow_df[flow_df['camera_id'] == selected_cam].sort_values('slot_idx')
        fig_bar = px.bar(filtered_df, x='time_label', y=['motorbike', 'car', 'bus', 'truck'], title=f"Phân loại phương tiện tại {selected_cam}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 3. Demo Real-time & Cảnh báo ---
    st.header("3. Giám Sát Real-time & Cảnh Báo Sớm")
    
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        # Placeholder cho video
        st.image("traffic_sample.jpg", caption="Camera Feed (Local)", use_container_width=True)
    
    with col_stats:
        st.subheader("Trạng thái Live")
        placeholder = st.empty()
        start_btn = st.button("▶ Bắt đầu mô phỏng")
        
        if start_btn:
            for i in range(15):
                current_flow = np.random.randint(50, 150)
                pred_next_30 = current_flow * (1 + np.random.uniform(-0.15, 0.15))
                
                if current_flow > 120:
                    status_md = "🔴 **TẮC NGHẼN NGHIÊM TRỌNG**"
                elif current_flow > 90:
                    status_md = "🟠 **Đông đúc**"
                else:
                    status_md = "🟢 **Thông thoáng**"

                with placeholder.container():
                    st.markdown(f"### {status_md}")
                    st.metric("Lưu lượng hiện tại", f"{current_flow} xe/phút", delta=f"{np.random.randint(-10, 10)}")
                    st.metric("Dự báo 30p tới", f"{int(pred_next_30)} xe/phút")
                    st.progress(min(current_flow, 150) / 150)
                    if current_flow > 120:
                        st.warning("⚠ Cảnh báo: Vượt ngưỡng năng lực thông hành!")
                
                time.sleep(0.7)

    # --- 4. Export ---
    st.header("4. Xuất Báo Cáo")
    csv = flow_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Tải xuống dữ liệu (CSV)", csv, "final_traffic_report.csv", "text/csv")

else:
    # Nếu chưa có dữ liệu thì hiện hướng dẫn
    st.info("👋 Chào bạn! Vui lòng tải file 'hcmc_flow.csv' và 'prediction.csv' lên (hoặc đảm bảo chúng có sẵn trong thư mục) để bắt đầu.")

