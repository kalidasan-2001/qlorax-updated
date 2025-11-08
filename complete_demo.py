#!/usr/bin/env python3
"""
🎯 QLORAX Complete Demo
Comprehensive demonstration of fine-tuned model capabilities
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time


class QLORAXDemo:
    def __init__(self, model_path="models/production-model/checkpoints"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.demo_queries = [
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

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("🔄 Loading fine-tuned model...")
        print(f"📁 Model path: {self.model_path}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("✅ Tokenizer loaded successfully")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=False,
            )
            self.model.eval()
            print("✅ Model loaded successfully")
            print(f"📊 Model parameters: {self.model.num_parameters():,}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Trying to load base model with adapters...")

            try:
                # Try loading as PEFT model
                base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"🔄 Loading base model: {base_model}")

                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype=torch.float32, device_map="auto"
                )

                # Load PEFT adapters
                self.model = PeftModel.from_pretrained(base_model_obj, self.model_path)
                self.model.eval()
                print("✅ PEFT model loaded successfully")
                return True

            except Exception as e2:
                print(f"❌ Error loading PEFT model: {e2}")
                return False

    def generate_response(
        self, prompt, max_length=200, temperature=0.7, do_sample=True
    ):
        """Generate response for a given prompt"""
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded"

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
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            inference_time = time.time() - start_time

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part
            response = full_response[len(formatted_prompt) :].strip()

            return {
                "response": response,
                "inference_time_ms": inference_time * 1000,
                "input_tokens": inputs.shape[1],
                "output_tokens": outputs.shape[1] - inputs.shape[1],
            }

        except Exception as e:
            return f"❌ Generation error: {e}"

    def run_interactive_demo(self):
        """Run interactive demo session"""
        print("\n🎮 Interactive Demo Mode")
        print("=" * 50)
        print("Ask questions about machine learning!")
        print("Type 'quit' to exit, 'demo' for predefined examples")
        print("=" * 50)

        while True:
            user_input = input("\n❓ Your question: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Thanks for trying QLORAX!")
                break
            elif user_input.lower() == "demo":
                self.run_predefined_demo()
                continue
            elif not user_input:
                continue

            print(f"\n🤔 Thinking about: '{user_input}'")
            result = self.generate_response(user_input)

            if isinstance(result, dict):
                print(f"\n🤖 Response:")
                print(f"   {result['response']}")
                print(
                    f"\n📊 Stats: {result['inference_time_ms']:.1f}ms | "
                    f"{result['input_tokens']} → {result['output_tokens']} tokens"
                )
            else:
                print(f"\n❌ {result}")

    def run_predefined_demo(self):
        """Run demo with predefined queries"""
        print("\n🎯 Predefined Demo Queries")
        print("=" * 50)

        results = []

        for i, query in enumerate(self.demo_queries, 1):
            print(f"\n📝 Query {i}/{len(self.demo_queries)}: {query}")
            result = self.generate_response(query)

            if isinstance(result, dict):
                print(f"🤖 Response: {result['response'][:100]}...")
                print(f"⚡ Time: {result['inference_time_ms']:.1f}ms")
                results.append(
                    {"query": query, "response": result["response"], "stats": result}
                )
            else:
                print(f"❌ Error: {result}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"demo_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Demo results saved to: {results_file}")
        return results

    def benchmark_performance(self):
        """Run performance benchmark"""
        print("\n⚡ Performance Benchmark")
        print("=" * 50)

        test_prompts = [
            "What is AI?",
            "Explain neural networks",
            "How does machine learning work?",
            "What is deep learning?",
            "Describe gradient descent",
        ]

        times = []
        token_counts = []

        for prompt in test_prompts:
            result = self.generate_response(prompt, max_length=100)
            if isinstance(result, dict):
                times.append(result["inference_time_ms"])
                token_counts.append(result["output_tokens"])

        if times:
            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)
            tokens_per_sec = (avg_tokens / avg_time) * 1000

            print(f"📊 Average inference time: {avg_time:.1f}ms")
            print(f"📝 Average output tokens: {avg_tokens:.1f}")
            print(f"🚀 Tokens per second: {tokens_per_sec:.1f}")
            print(f"🔬 Tested on {len(test_prompts)} queries")

        return {"avg_time_ms": avg_time, "tokens_per_sec": tokens_per_sec}

    def compare_responses(self):
        """Compare fine-tuned vs base model responses"""
        print("\n🔬 Model Comparison Demo")
        print("=" * 50)
        print("Comparing fine-tuned model responses...")

        # This would require loading base model too - simplified for demo
        test_query = "What is machine learning?"
        result = self.generate_response(test_query)

        print(f"📝 Query: {test_query}")
        if isinstance(result, dict):
            print(f"🎯 Fine-tuned Response:")
            print(f"   {result['response']}")

        print("\n💡 Note: Full comparison requires loading base model")

    def save_demo_report(self):
        """Generate and save demo report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"qlorax_demo_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write(f"# QLORAX Demo Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(f"**Model Path:** {self.model_path}\n\n")

            # Run quick benchmark for report
            perf = self.benchmark_performance()
            f.write(f"## Performance Metrics\n\n")
            f.write(f"- Average inference time: {perf.get('avg_time_ms', 0):.1f}ms\n")
            f.write(f"- Tokens per second: {perf.get('tokens_per_sec', 0):.1f}\n\n")

            # Sample responses
            f.write(f"## Sample Responses\n\n")
            for i, query in enumerate(self.demo_queries[:3], 1):
                result = self.generate_response(query, max_length=150)
                if isinstance(result, dict):
                    f.write(f"### Query {i}: {query}\n\n")
                    f.write(f"**Response:** {result['response']}\n\n")
                    f.write(
                        f"**Stats:** {result['inference_time_ms']:.1f}ms, {result['output_tokens']} tokens\n\n"
                    )

        print(f"📄 Demo report saved to: {report_file}")
        return report_file


def main():
    """Main demo function"""
    print("🎯 QLORAX Complete Demo")
    print("=" * 60)
    print("Welcome to the comprehensive QLORAX demonstration!")
    print("This will showcase your fine-tuned model's capabilities.")
    print("=" * 60)

    # Initialize demo
    demo = QLORAXDemo()

    # Load model
    if not demo.load_model():
        print("❌ Failed to load model. Please ensure training completed successfully.")
        return 1

    print("\n🎉 Model loaded successfully! Choose demo mode:")
    print("1. Interactive Q&A")
    print("2. Predefined demo queries")
    print("3. Performance benchmark")
    print("4. Generate demo report")
    print("5. Run all demos")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        demo.run_interactive_demo()
    elif choice == "2":
        demo.run_predefined_demo()
    elif choice == "3":
        demo.benchmark_performance()
    elif choice == "4":
        demo.save_demo_report()
    elif choice == "5":
        print("\n🚀 Running complete demo suite...")
        demo.run_predefined_demo()
        demo.benchmark_performance()
        demo.compare_responses()
        demo.save_demo_report()
        print("\n🎉 Complete demo finished!")
    else:
        print("❌ Invalid choice. Running interactive demo by default.")
        demo.run_interactive_demo()

    return 0


if __name__ == "__main__":
    exit(main())
