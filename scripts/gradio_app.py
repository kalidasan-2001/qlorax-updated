#!/usr/bin/env python3
"""
Gradio interface for QLORAX model interaction.
This provides a user-friendly web interface for testing your fine-tuned model.
"""

import gradio as gr
import requests
import json
from pathlib import Path

# Global variables for model (placeholder until model is trained)
model = None
tokenizer = None


def load_model_if_available():
    """Try to load the fine-tuned model if it exists."""
    model_path = Path("./models/tinyllama-qlora")

    if model_path.exists():
        try:
            # Uncomment when you have a trained model:
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # global model, tokenizer
            # model = AutoModelForCausalLM.from_pretrained(model_path)
            # tokenizer = AutoTokenizer.from_pretrained(model_path)
            # return True
            pass
        except Exception as e:
            print(f"Could not load model: {e}")

    return False


def generate_response(prompt, max_length, temperature, top_p):
    """Generate a response using the model."""

    # Check if model is loaded
    if model is None:
        # Placeholder response
        return f"🤖 [MOCK RESPONSE]\n\nPrompt: {prompt}\n\nResponse: This is a placeholder response! Once you train your model with Axolotl, this will show real AI-generated text based on your fine-tuning. The model would use your QLoRA fine-tuned TinyLlama to generate contextually relevant responses.\n\nModel Settings:\n- Max Length: {max_length}\n- Temperature: {temperature}\n- Top-p: {top_p}\n\n✨ Train your model to see real results!"

    # Real model inference (uncomment when ready)
    # try:
    #     inputs = tokenizer.encode(prompt, return_tensors="pt")
    #
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             inputs,
    #             max_length=max_length,
    #             temperature=temperature,
    #             top_p=top_p,
    #             do_sample=True,
    #             pad_token_id=tokenizer.eos_token_id
    #         )
    #
    #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return generated_text
    #
    # except Exception as e:
    #     return f"❌ Error generating response: {str(e)}"


def get_model_info():
    """Get information about the current model."""
    if model is None:
        return """
        📋 **Model Status**: No model loaded (using placeholder responses)
        
        🎯 **Expected Model**: TinyLlama-1.1B-Chat-v1.0 with QLoRA fine-tuning
        
        ⚙️ **Configuration**:
        - Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
        - Fine-tuning: QLoRA (Quantized LoRA) 
        - Quantization: 4-bit NF4
        - LoRA Rank: 64, Alpha: 16
        - Target Layers: q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj
        
        🚀 **To activate**: Train your model using `axolotl train configs/default-qlora-config.yml`
        """
    else:
        return f"""
        ✅ **Model Status**: Loaded and ready
        
        📊 **Model Details**:
        - Vocabulary Size: {tokenizer.vocab_size if tokenizer else 'Unknown'}
        - Max Length: {tokenizer.model_max_length if tokenizer else 'Unknown'}
        - Model Type: Causal Language Model with QLoRA
        """


def create_interface():
    """Create the Gradio interface."""

    # Load model if available
    model_loaded = load_model_if_available()

    with gr.Blocks(title="QLORAX - QLoRA Fine-tuned Model Interface") as demo:

        gr.Markdown(
            """
        # 🚀 QLORAX - QLoRA Fine-tuning Interface
        
        Welcome to your QLoRA fine-tuned model interface! This demo allows you to interact with your fine-tuned TinyLlama model.
        """
        )

        with gr.Tab("💬 Chat with Model"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Ask me anything about machine learning or QLoRA...",
                        lines=3,
                    )

                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=150,
                            step=10,
                            label="Max Length",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p",
                        )

                    generate_btn = gr.Button("🎯 Generate Response", variant="primary")

                with gr.Column(scale=3):
                    response_output = gr.Textbox(
                        label="Model Response", lines=15, max_lines=20
                    )

            # Example prompts
            gr.Examples(
                examples=[
                    ["What is machine learning?"],
                    [
                        "Explain the concept of fine-tuning in the context of large language models."
                    ],
                    ["What are the benefits of using QLoRA for model fine-tuning?"],
                    ["How does quantization help in reducing memory usage?"],
                    ["What is the difference between LoRA and full fine-tuning?"],
                ],
                inputs=prompt_input,
            )

            generate_btn.click(
                fn=generate_response,
                inputs=[prompt_input, max_length, temperature, top_p],
                outputs=response_output,
            )

        with gr.Tab("📊 Model Information"):
            model_info_output = gr.Markdown(value=get_model_info())
            refresh_btn = gr.Button("🔄 Refresh Model Info")
            refresh_btn.click(fn=get_model_info, outputs=model_info_output)

        with gr.Tab("📚 About QLoRA"):
            gr.Markdown(
                """
            ## 🎯 About QLoRA (Quantized LoRA)
            
            QLoRA is an efficient fine-tuning approach that combines:
            
            ### 🔧 **Key Features**:
            - **4-bit Quantization**: Reduces memory usage by ~75%
            - **LoRA (Low-Rank Adaptation)**: Only trains a small subset of parameters
            - **Double Quantization**: Further memory optimization
            - **NF4 Data Type**: Optimal quantization for neural networks
            
            ### 🎨 **Benefits**:
            - ✅ Fine-tune large models on consumer hardware
            - ✅ Faster training compared to full fine-tuning
            - ✅ Maintains model quality
            - ✅ Easy to swap adapters for different tasks
            
            ### 🔬 **Technical Details**:
            - **LoRA Rank**: 64 (number of trainable parameters)
            - **LoRA Alpha**: 16 (scaling factor)
            - **Target Modules**: attention and MLP layers
            - **Quantization**: 4-bit NF4 with double quantization
            
            ### 📖 **Resources**:
            - [QLoRA Paper](https://arxiv.org/abs/2305.14314)
            - [Axolotl Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
            - [Hugging Face PEFT](https://huggingface.co/docs/peft)
            """
            )

    return demo


def main():
    """Main function to launch the Gradio interface."""
    print("🚀 Starting QLORAX Gradio Interface...")

    demo = create_interface()

    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
    )


if __name__ == "__main__":
    main()
