#!/usr/bin/env python3
"""
Simple Web Demo for QLORAX Trained Model
Works with LoRA adapters
"""

import warnings

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


class SimpleQLORAXDemo:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the LoRA fine-tuned model"""
        try:
            print("Loading base model and tokenizer...")
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float32, device_map="cpu"
            )

            print("Loading LoRA adapter...")
            adapter_path = "models/production-model/checkpoints"
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.eval()
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def generate_response(self, user_input, max_length=150, temperature=0.7):
        """Generate response from the fine-tuned model"""
        if self.model is None:
            return "❌ Model not loaded properly. Please check the console for errors."

        if not user_input.strip():
            return "Please enter a question or prompt."

        try:
            # Format input using the training template
            prompt = f"### Input:\n{user_input.strip()}\n\n### Output:\n"

            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )

            # Decode and extract response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the output part
            if "### Output:\n" in full_response:
                response = full_response.split("### Output:\n")[-1].strip()
            else:
                response = full_response

            return response

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"


# Initialize the demo
demo_instance = SimpleQLORAXDemo()


def chat_fn(message, history, max_length, temperature):
    """Chat function for Gradio interface"""
    response = demo_instance.generate_response(message, max_length, temperature)
    return response


# Create Gradio interface
with gr.Blocks(title="QLORAX Fine-tuned Model Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # 🚀 QLORAX Fine-tuned Model Demo
    
    This is your **QLoRA fine-tuned TinyLlama model** trained on custom data.
    Ask questions and see how your model responds!
    
    ## ✨ What this model knows:
    - Machine learning concepts
    - Fine-tuning techniques
    - QLoRA methodology
    - General knowledge from training data
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask me about machine learning, QLoRA, or anything...",
                    lines=3,
                )

                with gr.Row():
                    submit_btn = gr.Button("🤖 Generate Response", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                max_length = gr.Slider(
                    minimum=50,
                    maximum=300,
                    value=150,
                    step=10,
                    label="Max Response Length",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (Creativity)",
                )

    with gr.Group():
        response_output = gr.Textbox(
            label="Model Response",
            lines=6,
            placeholder="Response will appear here...",
            interactive=False,
        )

    # Event handlers
    submit_btn.click(
        fn=lambda msg, max_len, temp: demo_instance.generate_response(
            msg, max_len, temp
        ),
        inputs=[user_input, max_length, temperature],
        outputs=[response_output],
    )

    clear_btn.click(fn=lambda: ("", ""), outputs=[user_input, response_output])

    user_input.submit(
        fn=lambda msg, max_len, temp: demo_instance.generate_response(
            msg, max_len, temp
        ),
        inputs=[user_input, max_length, temperature],
        outputs=[response_output],
    )

    gr.Markdown(
        """
    ### 🎯 Try these example questions:
    - "What is machine learning?"
    - "Explain QLoRA fine-tuning"
    - "What are the benefits of using LoRA?"
    - "How does fine-tuning work?"
    
    ---
    **Model**: TinyLlama-1.1B-Chat-v1.0 + QLoRA  
    **Training**: CPU-optimized production configuration  
    **Framework**: QLORAX Enhanced
    """
    )

if __name__ == "__main__":
    print("🌐 Starting QLORAX Web Demo...")
    print("🔗 Access the demo at: http://localhost:7860")
    demo.launch(server_name="localhost", server_port=7860, share=False, debug=False)
