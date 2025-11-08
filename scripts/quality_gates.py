#!/usr/bin/env python3
"""
🎯 QLORAX Quality Gates Implementation
Automated evaluation criteria matching the CI/CD pipeline decision points
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QualityGateEvaluator:
    """Quality gate evaluator for QLORAX CI/CD pipeline"""

    def __init__(self, config_path: str = "configs/quality-gates.json"):
        """Initialize quality gate evaluator

        Args:
            config_path: Path to quality gates configuration
        """
        self.config_path = Path(config_path)
        self.config = self.load_quality_gates_config()
        self.evaluation_results = {}

    def load_quality_gates_config(self) -> Dict[str, Any]:
        """Load quality gates configuration"""
        default_config = {
            "thresholds": {
                "bert_f1_minimum": 0.90,
                "rouge_l_minimum": 0.85,
                "coherence_minimum": 0.85,
                "response_time_maximum": 2.0,
                "model_size_maximum_mb": 50,
                "training_loss_maximum": 2.0,
            },
            "gates": {
                "dry_run": {
                    "required_checks": [
                        "system_validation",
                        "integration_tests",
                        "config_validation",
                    ],
                    "allow_failure": False,
                },
                "training": {
                    "required_metrics": ["training_loss", "model_size"],
                    "allow_partial_failure": False,
                },
                "evaluation": {
                    "required_metrics": ["bert_f1", "rouge_l", "coherence_score"],
                    "critical_metrics": ["bert_f1", "rouge_l"],
                    "allow_partial_failure": True,
                },
                "deployment": {
                    "required_checks": ["model_loading", "api_health", "response_time"],
                    "allow_failure": False,
                },
            },
            "scoring": {
                "weights": {
                    "bert_f1": 0.4,
                    "rouge_l": 0.3,
                    "coherence": 0.2,
                    "efficiency": 0.1,
                },
                "grade_thresholds": {
                    "A+": 0.95,
                    "A": 0.90,
                    "B+": 0.85,
                    "B": 0.80,
                    "C": 0.70,
                    "F": 0.0,
                },
            },
        }

        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        else:
            # Create default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def evaluate_dry_run_gate(
        self, validation_results: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate dry run quality gate

        Args:
            validation_results: Results from system validation and tests

        Returns:
            Tuple of (passed, details)
        """
        gate_config = self.config["gates"]["dry_run"]
        required_checks = gate_config["required_checks"]

        results = {
            "gate_name": "dry_run",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "passed": True,
            "details": [],
        }

        # Check required validation results
        for check in required_checks:
            if check in validation_results and validation_results[check]["passed"]:
                results["checks"][check] = {
                    "status": "PASS",
                    "details": validation_results[check],
                }
                results["details"].append(f"✅ {check}: PASSED")
            else:
                results["checks"][check] = {
                    "status": "FAIL",
                    "details": validation_results.get(check, {}),
                }
                results["details"].append(f"❌ {check}: FAILED")
                results["passed"] = False

        # Additional validation: Config file validity
        if "config_validation" in required_checks:
            config_valid = self.validate_training_config()
            if config_valid:
                results["details"].append("✅ Training configuration: VALID")
            else:
                results["details"].append("❌ Training configuration: INVALID")
                results["passed"] = False

        logger.info(
            f"Dry run gate evaluation: {'PASSED' if results['passed'] else 'FAILED'}"
        )
        return results["passed"], results

    def evaluate_training_gate(
        self, training_results: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate training quality gate

        Args:
            training_results: Results from model training

        Returns:
            Tuple of (passed, details)
        """
        gate_config = self.config["gates"]["training"]
        thresholds = self.config["thresholds"]

        results = {
            "gate_name": "training",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "passed": True,
            "details": [],
        }

        # Check training loss
        if "final_loss" in training_results:
            loss = training_results["final_loss"]
            loss_threshold = thresholds["training_loss_maximum"]

            if loss <= loss_threshold:
                results["metrics"]["training_loss"] = {
                    "value": loss,
                    "status": "PASS",
                    "threshold": loss_threshold,
                }
                results["details"].append(
                    f"✅ Training loss: {loss:.4f} ≤ {loss_threshold}"
                )
            else:
                results["metrics"]["training_loss"] = {
                    "value": loss,
                    "status": "FAIL",
                    "threshold": loss_threshold,
                }
                results["details"].append(
                    f"❌ Training loss: {loss:.4f} > {loss_threshold}"
                )
                results["passed"] = False

        # Check model size
        if "model_size_mb" in training_results:
            size = training_results["model_size_mb"]
            size_threshold = thresholds["model_size_maximum_mb"]

            if size <= size_threshold:
                results["metrics"]["model_size"] = {
                    "value": size,
                    "status": "PASS",
                    "threshold": size_threshold,
                }
                results["details"].append(
                    f"✅ Model size: {size:.1f}MB ≤ {size_threshold}MB"
                )
            else:
                results["metrics"]["model_size"] = {
                    "value": size,
                    "status": "FAIL",
                    "threshold": size_threshold,
                }
                results["details"].append(
                    f"❌ Model size: {size:.1f}MB > {size_threshold}MB"
                )
                results["passed"] = False

        # Check training completion
        if (
            "training_completed" in training_results
            and not training_results["training_completed"]
        ):
            results["details"].append("❌ Training did not complete successfully")
            results["passed"] = False

        logger.info(
            f"Training gate evaluation: {'PASSED' if results['passed'] else 'FAILED'}"
        )
        return results["passed"], results

    def evaluate_evaluation_gate(
        self, eval_results: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate model evaluation quality gate

        Args:
            eval_results: Results from model evaluation

        Returns:
            Tuple of (passed, details)
        """
        gate_config = self.config["gates"]["evaluation"]
        thresholds = self.config["thresholds"]

        results = {
            "gate_name": "evaluation",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "passed": True,
            "critical_failed": False,
            "details": [],
            "overall_score": 0.0,
            "grade": "F",
        }

        # Extract metrics from evaluation results
        metrics = eval_results.get("metrics", {})

        # Check BERT F1 Score (Critical)
        if "bert_scores" in metrics and "bert_f1" in metrics["bert_scores"]:
            bert_f1 = metrics["bert_scores"]["bert_f1"]
            threshold = thresholds["bert_f1_minimum"]

            if bert_f1 >= threshold:
                results["metrics"]["bert_f1"] = {
                    "value": bert_f1,
                    "status": "PASS",
                    "threshold": threshold,
                }
                results["details"].append(f"✅ BERT F1: {bert_f1:.4f} ≥ {threshold}")
            else:
                results["metrics"]["bert_f1"] = {
                    "value": bert_f1,
                    "status": "FAIL",
                    "threshold": threshold,
                }
                results["details"].append(f"❌ BERT F1: {bert_f1:.4f} < {threshold}")
                results["critical_failed"] = True

        # Check ROUGE-L Score (Critical)
        if "rouge_scores" in metrics and "rougeL" in metrics["rouge_scores"]:
            rouge_l = metrics["rouge_scores"]["rougeL"]
            threshold = thresholds["rouge_l_minimum"]

            if rouge_l >= threshold:
                results["metrics"]["rouge_l"] = {
                    "value": rouge_l,
                    "status": "PASS",
                    "threshold": threshold,
                }
                results["details"].append(f"✅ ROUGE-L: {rouge_l:.4f} ≥ {threshold}")
            else:
                results["metrics"]["rouge_l"] = {
                    "value": rouge_l,
                    "status": "FAIL",
                    "threshold": threshold,
                }
                results["details"].append(f"❌ ROUGE-L: {rouge_l:.4f} < {threshold}")
                results["critical_failed"] = True

        # Check Coherence Score (Non-critical)
        if (
            "quality_metrics" in metrics
            and "coherence_score" in metrics["quality_metrics"]
        ):
            coherence = metrics["quality_metrics"]["coherence_score"]
            threshold = thresholds["coherence_minimum"]

            if coherence >= threshold:
                results["metrics"]["coherence"] = {
                    "value": coherence,
                    "status": "PASS",
                    "threshold": threshold,
                }
                results["details"].append(
                    f"✅ Coherence: {coherence:.4f} ≥ {threshold}"
                )
            else:
                results["metrics"]["coherence"] = {
                    "value": coherence,
                    "status": "WARN",
                    "threshold": threshold,
                }
                results["details"].append(
                    f"⚠️  Coherence: {coherence:.4f} < {threshold}"
                )

        # Calculate overall score and grade
        overall_score = self.calculate_overall_score(results["metrics"])
        results["overall_score"] = overall_score
        results["grade"] = self.assign_grade(overall_score)

        # Determine if gate passes
        results["passed"] = not results["critical_failed"] and overall_score >= 0.80

        results["details"].append(f"📊 Overall Score: {overall_score:.4f}")
        results["details"].append(f"🎯 Grade: {results['grade']}")

        logger.info(
            f"Evaluation gate: {'PASSED' if results['passed'] else 'FAILED'} (Grade: {results['grade']})"
        )
        return results["passed"], results

    def evaluate_deployment_gate(
        self, deployment_results: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate deployment readiness quality gate

        Args:
            deployment_results: Results from deployment checks

        Returns:
            Tuple of (passed, details)
        """
        gate_config = self.config["gates"]["deployment"]
        thresholds = self.config["thresholds"]

        results = {
            "gate_name": "deployment",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "passed": True,
            "details": [],
        }

        # Check model loading
        if "model_loading" in deployment_results:
            loading_success = deployment_results["model_loading"]["success"]
            loading_time = deployment_results["model_loading"].get("time_seconds", 0)

            if loading_success:
                results["checks"]["model_loading"] = {
                    "status": "PASS",
                    "time": loading_time,
                }
                results["details"].append(
                    f"✅ Model loading: SUCCESS ({loading_time:.2f}s)"
                )
            else:
                results["checks"]["model_loading"] = {
                    "status": "FAIL",
                    "error": deployment_results["model_loading"].get("error"),
                }
                results["details"].append(f"❌ Model loading: FAILED")
                results["passed"] = False

        # Check API health
        if "api_health" in deployment_results:
            api_healthy = deployment_results["api_health"]["healthy"]
            response_code = deployment_results["api_health"].get("response_code")

            if api_healthy and response_code == 200:
                results["checks"]["api_health"] = {
                    "status": "PASS",
                    "response_code": response_code,
                }
                results["details"].append(
                    f"✅ API health: HEALTHY (HTTP {response_code})"
                )
            else:
                results["checks"]["api_health"] = {
                    "status": "FAIL",
                    "response_code": response_code,
                }
                results["details"].append(
                    f"❌ API health: UNHEALTHY (HTTP {response_code})"
                )
                results["passed"] = False

        # Check response time
        if "response_time" in deployment_results:
            avg_response_time = deployment_results["response_time"]["average_seconds"]
            threshold = thresholds["response_time_maximum"]

            if avg_response_time <= threshold:
                results["checks"]["response_time"] = {
                    "status": "PASS",
                    "value": avg_response_time,
                    "threshold": threshold,
                }
                results["details"].append(
                    f"✅ Response time: {avg_response_time:.2f}s ≤ {threshold}s"
                )
            else:
                results["checks"]["response_time"] = {
                    "status": "FAIL",
                    "value": avg_response_time,
                    "threshold": threshold,
                }
                results["details"].append(
                    f"❌ Response time: {avg_response_time:.2f}s > {threshold}s"
                )
                results["passed"] = False

        logger.info(
            f"Deployment gate evaluation: {'PASSED' if results['passed'] else 'FAILED'}"
        )
        return results["passed"], results

    def calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall score"""
        weights = self.config["scoring"]["weights"]
        total_score = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():
            if metric_name == "bert_f1" and "bert_f1" in metrics:
                total_score += metrics["bert_f1"]["value"] * weight
                total_weight += weight
            elif metric_name == "rouge_l" and "rouge_l" in metrics:
                total_score += metrics["rouge_l"]["value"] * weight
                total_weight += weight
            elif metric_name == "coherence" and "coherence" in metrics:
                total_score += metrics["coherence"]["value"] * weight
                total_weight += weight
            elif metric_name == "efficiency":
                # Calculate efficiency based on model size and response time
                efficiency = self.calculate_efficiency_score(metrics)
                total_score += efficiency * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on model size and response time"""
        # This is a simplified efficiency calculation
        # In practice, you might want more sophisticated metrics
        base_score = 0.8  # Base efficiency score

        # Adjust based on available metrics
        if "model_size" in metrics:
            size_mb = metrics["model_size"]["value"]
            if size_mb <= 20:
                base_score += 0.1  # Bonus for small models
            elif size_mb >= 50:
                base_score -= 0.1  # Penalty for large models

        return min(1.0, max(0.0, base_score))

    def assign_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        thresholds = self.config["scoring"]["grade_thresholds"]

        for grade, threshold in thresholds.items():
            if score >= threshold:
                return grade

        return "F"

    def validate_training_config(self) -> bool:
        """Validate training configuration files"""
        required_configs = [
            "configs/production-config.yaml",
            "configs/instructlab-config.yaml",
        ]

        for config_path in required_configs:
            if not Path(config_path).exists():
                logger.warning(f"Missing config file: {config_path}")
                return False

        return True

    def save_quality_gate_results(self, results: Dict[str, Any], output_path: str):
        """Save quality gate results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Quality gate results saved to: {output_file}")


def main():
    """Main quality gate evaluation function"""
    parser = argparse.ArgumentParser(description="QLORAX Quality Gate Evaluator")
    parser.add_argument(
        "--gate",
        choices=["dry-run", "training", "evaluation", "deployment"],
        required=True,
        help="Quality gate to evaluate",
    )
    parser.add_argument("--results-file", required=True, help="Path to results file")
    parser.add_argument(
        "--output", default="results/quality-gate.json", help="Output file for results"
    )
    parser.add_argument(
        "--config",
        default="configs/quality-gates.json",
        help="Quality gates configuration",
    )

    args = parser.parse_args()

    # Load results
    with open(args.results_file, "r") as f:
        results = json.load(f)

    # Initialize evaluator
    evaluator = QualityGateEvaluator(args.config)

    # Evaluate specified gate
    if args.gate == "dry-run":
        passed, gate_results = evaluator.evaluate_dry_run_gate(results)
    elif args.gate == "training":
        passed, gate_results = evaluator.evaluate_training_gate(results)
    elif args.gate == "evaluation":
        passed, gate_results = evaluator.evaluate_evaluation_gate(results)
    elif args.gate == "deployment":
        passed, gate_results = evaluator.evaluate_deployment_gate(results)

    # Save results
    evaluator.save_quality_gate_results(gate_results, args.output)

    # Print summary
    print(f"\n🎯 Quality Gate: {args.gate.upper()}")
    print(f"{'='*50}")
    for detail in gate_results["details"]:
        print(f"   {detail}")
    print(f"{'='*50}")
    print(f"Result: {'✅ PASSED' if passed else '❌ FAILED'}")

    # Exit with appropriate code
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
