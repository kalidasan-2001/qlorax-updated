#!/usr/bin/env python3
"""
🌐 QLORAX Web Demo
Interactive web interface for your fine-tuned model
"""

import json
import time
import warnings
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class QLORAXWebDemo:
    def __init__(self, model_path="models/production-model"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model"""
        try:
            print(f"🔄 Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token or "[PAD]"
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=torch.float32, device_map="auto"
            )
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try loading as PEFT model
            try:
                base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype=torch.float32, device_map="auto"
                )
                self.model = PeftModel.from_pretrained(base_model_obj, self.model_path)
                self.model.eval()
                print("✅ PEFT model loaded successfully")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                self.model = None
                self.tokenizer = None

    def generate_response(self, prompt, max_length=200, temperature=0.7, top_p=0.9):
        """Generate response for web interface"""
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded. Please check the model path.", ""

        try:
            # Format prompt
            formatted_prompt = f"### Input:\n{prompt}\n\n### Output:\n"

            # Tokenize
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            inference_time = time.time() - start_time

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(formatted_prompt) :].strip()

            # Stats
            stats = f"⚡ Generated in {inference_time*1000:.1f}ms | {outputs.shape[1] - inputs.shape[1]} tokens"

            return response, stats

        except Exception as e:
            return f"❌ Generation error: {e}", ""


def create_demo_interface():
    """Create Gradio interface"""
    demo_instance = QLORAXWebDemo()

    # Predefined examples
    examples = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain gradient descent in simple terms",
        "What is the difference between supervised and unsupervised learning?",
        "How does backpropagation work?",
        "What are the advantages of deep learning?",
        "Explain overfitting and how to prevent it",
        "What is cross-validation?",
        "How do you evaluate a machine learning model?",
        "What is the bias-variance tradeoff?",
    ]

    def chat_fn(message, history, max_length, temperature, top_p):
        """Chat function for gradio"""
        response, stats = demo_instance.generate_response(
            message, max_length, temperature, top_p
        )

        # Add to history
        history.append([message, response])

        return history, "", stats

    def clear_fn():
        """Clear chat history"""
        return [], ""

    # Create interface
    with gr.Blocks(
        title="QLORAX Fine-tuned Model Demo", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown(
            """
        # 🎯 QLORAX Fine-tuned Model Demo
        
        **Welcome to your fine-tuned language model!** This model has been trained on machine learning Q&A data using QLoRA (Quantized Low-Rank Adaptation).
        
        Ask questions about machine learning, AI, data science, and related topics to see how well your model performs!
        """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat with your fine-tuned model", height=400, show_label=True
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="Ask anything about machine learning...",
                        lines=2,
                        scale=4,
                    )
                    submit = gr.Button("🚀 Ask", scale=1, variant="primary")

                with gr.Row():
                    clear = gr.Button("🗑️ Clear Chat", scale=1)
                    stats = gr.Textbox(
                        label="Generation Stats", scale=3, interactive=False
                    )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Generation Settings")

                max_length = gr.Slider(
                    minimum=50, maximum=500, value=200, step=10, label="Max Length"
                )

                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"
                )

                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p"
                )

                gr.Markdown("### 📝 Try These Examples")
                example_dropdown = gr.Dropdown(
                    choices=examples, label="Select example", value=examples[0]
                )

                use_example = gr.Button("📋 Use Example", variant="secondary")

        # Event handlers
        submit.click(
            chat_fn,
            inputs=[msg, chatbot, max_length, temperature, top_p],
            outputs=[chatbot, msg, stats],
        )

        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, max_length, temperature, top_p],
            outputs=[chatbot, msg, stats],
        )

        clear.click(clear_fn, outputs=[chatbot, stats])

        use_example.click(lambda x: x, inputs=[example_dropdown], outputs=[msg])

        gr.Markdown(
            """
        ---
        ### 🎯 Model Information
        - **Base Model**: TinyLlama 1.1B Chat
        - **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)  
        - **Training Data**: Machine Learning Q&A Dataset
        - **Inference**: CPU-optimized for broad compatibility
        
        ### 💡 Tips for Better Results
        - Ask specific questions about ML concepts
        - Use clear, well-formed questions
        - Experiment with temperature settings (lower = more focused, higher = more creative)
        - Try different max lengths for shorter or longer responses
        """
        )

    return interface


def main():
    """Launch the web demo"""
    print("🌐 Starting QLORAX Web Demo...")

    # Create and launch interface
    demo = create_demo_interface()

    print("🚀 Launching web interface...")
    print("📱 The demo will open in your browser automatically")
    print("🔗 You can also access it at: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop the server")

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_api=False,
        inbrowser=True,  # Open browser automatically
    )


if __name__ == "__main__":
    main()
