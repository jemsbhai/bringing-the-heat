"""Local-only Gradio preview of the measured INT8 artifact. Not a public service."""
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
import torch
from demo import LABELS, MAX_LENGTH, load_backend

model, tokenizer = load_backend("int8")


def classify(text):
    if not text.strip():
        raise gr.Error("Enter a news headline or short article.")
    if len(text) > 10000:
        raise gr.Error("Please use at most 10,000 characters.")
    with torch.inference_mode():
        inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        scores = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: score for label, score in zip(LABELS, scores)}


app = gr.Interface(classify, gr.Textbox(lines=4, label="News text"),
    gr.Label(num_top_classes=4, label="Model scores"),
    title="Bringing the Heat | News Router",
    description="A local CPU demo: Hugging Face Hub → PEFT LoRA + Accelerate → Optimum ONNX INT8. Scores are not calibrated probabilities. Text is truncated to 128 tokens.",
    examples=["The Miami team won the championship after a dramatic final quarter.",
              "Shares climbed after the company reported higher quarterly earnings.",
              "Scientists unveiled a new processor for energy-efficient computing.",
              "Leaders met for talks on a new international peace agreement."],
    flagging_mode="never")

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
