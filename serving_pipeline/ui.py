import gradio as gr
import pandas as pd
import random
import time

# --- 1. LOGIC DỰ ĐOÁN (Cũ) ---

def predict_single(user_id, product_id, price, brand, activity_count, weekday):
    base_score = 0.2
    try: p = float(price)
    except: p = 500
    if p < 50: base_score += 0.3
    if activity_count > 5: base_score += 0.25
    if str(brand).lower() in ["samsung", "apple"]: base_score += 0.15
    score = base_score + (random.random() * 0.15)
    score = min(score, 0.99)
    is_buy = 1 if score > 0.5 else 0
    result_text = "CÓ MUA (Purchase)" if is_buy == 1 else "KHÔNG MUA (No Purchase)"
    return f"User: {user_id}\nProduct: {product_id}\nDự đoán: {result_text}\nXác suất: {score:.4f}"

def batch_predict(file_obj):
    if file_obj is None: return None
    try:
        df = pd.read_csv(file_obj.name)
        df['purchase_prob'] = [round(random.uniform(0.1, 0.99), 4) for _ in range(len(df))]
        df['is_purchased_pred'] = df['purchase_prob'].apply(lambda x: 1 if x > 0.5 else 0)
        return df
    except Exception as e:
        return pd.DataFrame({"Lỗi": [str(e)]})

# --- 2. LOGIC CHATBOT (Mới) ---

def call_ai_api(message, history):
    """
    Hàm xử lý tin nhắn và gọi API AI.
    Hiện tại đang giả lập phản hồi. Bạn hãy thay thế phần này bằng code gọi API thật.
    """
    # --- VÙNG TÍCH HỢP API (VD: OpenAI, Gemini) ---
    # import openai
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[...])
    # bot_message = response.choices[0].message.content
    # ----------------------------------------------
    
    # Giả lập độ trễ của API
    time.sleep(1)
    
    # Logic phản hồi giả lập (Mock Response)
    msg_lower = message.lower()
    if "xin chào" in msg_lower:
        return "Chào bạn! Tôi là trợ lý AI phân tích dữ liệu E-commerce. Tôi có thể giúp gì cho bạn?"
    elif "dự đoán" in msg_lower:
        return "Để dự đoán khả năng mua hàng, bạn vui lòng qua tab 'Single Prediction' hoặc upload file ở 'Batch Prediction' nhé."
    elif "code" in msg_lower:
        return "Hệ thống này được xây dựng bằng Python và thư viện Gradio, sử dụng mô hình XGBoost (giả định) để phân loại."
    else:
        return f"Tôi đã nhận được câu hỏi: '{message}'. Tuy nhiên, tôi chưa được kết nối với API thực tế. Vui lòng cấu hình API Key trong mã nguồn."

def chat_interface(message, history):
    # History là danh sách các cặp [user_msg, bot_msg]
    # Gradio Chatbot mong đợi history trả về để hiển thị
    bot_message = call_ai_api(message, history)
    history.append((message, bot_message))
    return "", history # Trả về rỗng cho ô nhập liệu, và history mới

# --- 3. GIAO DIỆN (UI) ---

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    body_text_color="white"
)

custom_css = """
body { background-color: #0B0F19; color: white; }
gradio-app { background-color: #0B0F19; }
.gradio-container { background-color: #0B0F19 !important; }
h1, h2, h3, p { color: white !important; }
.dataframe { color: white !important; }
/* Chatbot Customization */
#chatbot { height: 500px !important; overflow-y: auto; background-color: #1e293b; }
"""

with gr.Blocks(theme=theme, css=custom_css, title="E-commerce AI System") as demo:
    
    gr.Markdown("# E-commerce AI Prediction & Assistant")
    
    with gr.Tabs():
        
        # === TAB 1: SINGLE PREDICTION ===
        with gr.TabItem("Single Prediction"):
            with gr.Row():
                with gr.Column():
                    user_id = gr.Textbox(label="User ID")
                    product_id = gr.Textbox(label="Product ID")
                with gr.Column():
                    price = gr.Number(label="Price ($)", value=100.0)
                    brand = gr.Dropdown(choices=["samsung", "apple", "other"], value="other", label="Brand", allow_custom_value=True)
                with gr.Column():
                    act_count = gr.Slider(1, 50, 5, step=1, label="Activity Count")
                    weekday = gr.Dropdown(choices=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value="Sun", label="Weekday")
            
            btn_pred = gr.Button("Predict", variant="primary")
            out_pred = gr.Textbox(label="Result", interactive=False)
            btn_pred.click(predict_single, [user_id, product_id, price, brand, act_count, weekday], out_pred)

        # === TAB 2: BATCH PREDICTION ===
        with gr.TabItem("Batch Prediction"):
            with gr.Row():
                file_in = gr.File(label="Upload CSV")
                btn_batch = gr.Button("Run Batch", variant="primary")
            out_batch = gr.Dataframe(label="Results")
            btn_batch.click(batch_predict, file_in, out_batch)

        # === TAB 3: AI DATA ASSISTANT (NEW) ===
        with gr.TabItem("AI Chatbot"):
            gr.Markdown("### Trò chuyện với AI về dữ liệu (Integration Ready)")
            
            chatbot = gr.Chatbot(label="Hội thoại", elem_id="chatbot", bubble_full_width=False)
            msg = gr.Textbox(label="Nhập câu hỏi của bạn...", placeholder="Hỏi về xu hướng dữ liệu, giải thích model...", autofocus=True)
            
            with gr.Row():
                send_btn = gr.Button("Gửi tin nhắn", variant="primary")
                clear_btn = gr.Button("Xóa hội thoại")

            # Xử lý sự kiện: Nhấn Enter hoặc nút Gửi
            msg.submit(chat_interface, inputs=[msg, chatbot], outputs=[msg, chatbot])
            send_btn.click(chat_interface, inputs=[msg, chatbot], outputs=[msg, chatbot])
            
            # Nút xóa
            clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()