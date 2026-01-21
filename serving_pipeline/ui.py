import gradio as gr
import pandas as pd
import random
import time

# --- 1. LOGIC DỰ ĐOÁN ---

def predict_single(user_id, product_id, price, brand, activity_count, weekday):
    base_score = 0.2
    try:
        p = float(price)
    except:
        p = 500

    if p < 50:
        base_score += 0.3
    if activity_count > 5:
        base_score += 0.25
    if str(brand).lower() in ["samsung", "apple"]:
        base_score += 0.15

    score = base_score + (random.random() * 0.15)
    score = min(score, 0.99)
    is_buy = 1 if score > 0.5 else 0

    result_text = "CÓ MUA (Purchase)" if is_buy else "KHÔNG MUA (No Purchase)"
    return (
        f"User: {user_id}\n"
        f"Product: {product_id}\n"
        f"Dự đoán: {result_text}\n"
        f"Xác suất: {score:.4f}"
    )


def batch_predict(file_obj):
    if file_obj is None:
        return None
    try:
        df = pd.read_csv(file_obj.name)
        df["purchase_prob"] = [round(random.uniform(0.1, 0.99), 4) for _ in range(len(df))]
        df["is_purchased_pred"] = (df["purchase_prob"] > 0.5).astype(int)
        return df
    except Exception as e:
        return pd.DataFrame({"Lỗi": [str(e)]})


# --- 2. LOGIC CHATBOT ---

def call_ai_api(message, history):
    time.sleep(1)
    msg_lower = message.lower()

    if "xin chào" in msg_lower:
        return "Chào bạn! Tôi là trợ lý AI phân tích dữ liệu E-commerce."
    elif "dự đoán" in msg_lower:
        return "Bạn có thể dùng tab Single hoặc Batch Prediction."
    elif "code" in msg_lower:
        return "Hệ thống dùng Python + Gradio, model hiện đang mô phỏng."
    else:
        return f"Tôi đã nhận câu hỏi: '{message}'"


def chat_interface(message, history):
    bot_message = call_ai_api(message, history)
    history.append((message, bot_message))
    return "", history


# --- 3. UI ---

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    body_text_color="white",
)

custom_css = """
body { background-color: #0B0F19; color: white; }
.gradio-container { background-color: #0B0F19 !important; }
h1, h2, h3, p { color: white !important; }
#chatbot { height: 500px; overflow-y: auto; background-color: #1e293b; }
"""

# ✅ title, theme, css đặt ở Blocks (KHÔNG đặt ở launch)
with gr.Blocks(
    title="E-commerce AI System",
    theme=theme,
    css=custom_css
) as demo:

    gr.Markdown("# E-commerce AI Prediction & Assistant")

    with gr.Tabs():

        # === TAB 1 ===
        with gr.TabItem("Single Prediction"):
            with gr.Row():
                with gr.Column():
                    user_id = gr.Textbox(label="User ID")
                    product_id = gr.Textbox(label="Product ID")
                with gr.Column():
                    price = gr.Number(label="Price ($)", value=100.0)
                    brand = gr.Dropdown(
                        choices=["samsung", "apple", "other"],
                        value="other",
                        label="Brand",
                        allow_custom_value=True,
                    )
                with gr.Column():
                    act_count = gr.Slider(1, 50, 5, step=1, label="Activity Count")
                    weekday = gr.Dropdown(
                        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        value="Sun",
                        label="Weekday",
                    )

            btn_pred = gr.Button("Predict", variant="primary")
            out_pred = gr.Textbox(label="Result", interactive=False)
            btn_pred.click(
                predict_single,
                [user_id, product_id, price, brand, act_count, weekday],
                out_pred,
            )

        # === TAB 2 ===
        with gr.TabItem("Batch Prediction"):
            file_in = gr.File(label="Upload CSV")
            btn_batch = gr.Button("Run Batch", variant="primary")
            out_batch = gr.Dataframe(label="Results")
            btn_batch.click(batch_predict, file_in, out_batch)

        # === TAB 3 ===
        with gr.TabItem("AI Chatbot"):
            chatbot = gr.Chatbot(elem_id="chatbot")
            msg = gr.Textbox(placeholder="Hỏi về dữ liệu, model, xu hướng...")

            with gr.Row():
                send_btn = gr.Button("Gửi", variant="primary")
                clear_btn = gr.Button("Xóa")

            msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
            send_btn.click(chat_interface, [msg, chatbot], [msg, chatbot])
            clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch()
