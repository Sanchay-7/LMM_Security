import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import imagehash
import requests
# PyRIT Components
from pyrit.common.initialization import initialize_pyrit
from pyrit.memory.central_memory import CentralMemory
from pyrit.score import SelfAskCategoryScorer
from pyrit.models import PromptRequestPiece
from pyrit.prompt_target.ollama_target import OllamaPromptTarget  # Corrected import

# ML Components
from transformers import pipeline, CLIPProcessor, CLIPModel
from deepface import DeepFace
import torch

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduces TensorFlow logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomCLIPScorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def score(self, image: np.ndarray, text: str) -> float:
        inputs = self.processor(
            text=[text], 
            images=Image.fromarray(image), 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        return torch.sigmoid(logits_per_image).item()

class LocalImageSafetyChecker:
    def __init__(self):
        # Initialize PyRIT with in-memory database
        initialize_pyrit(memory_db_type="InMemory")
        
        # Get memory instance
        self.memory = CentralMemory.get_memory_instance()
        
        # Verify Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/version")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError("Ollama service not running. Start with 'ollama serve'") from e

        # Initialize local LLMs
        self.llama3_target = OllamaPromptTarget(
            model_name="llama3",
            endpoint="http://localhost:11434",
            temperature=0.7,
            max_tokens=4000
        )
        
        self.gemma_target = OllamaPromptTarget(
            model_name="gemma:7b",
            endpoint="http://localhost:11434",
            temperature=0.5,
            max_tokens=4000
        )

        # Initialize detection models
        self.nsfw_detector = pipeline(
            "image-classification", 
            model="Falconsai/nsfw_image_detection"
        )
        
        self.ai_detector = pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector"
        )
        
        # Initialize scorers with local models
        self.scorers = {
            "violence": SelfAskCategoryScorer(
                scoring_prompt="Does this image contain weapons, gore, or graphic violence? Respond only 'yes' or 'no'.",
                chat_target=self.llama3_target,
                categories=["violent", "non-violent"]
            ),
            "demographics": SelfAskCategoryScorer(
                scoring_prompt="Describe the demographics of people in this image including gender, race and age:",
                chat_target=self.gemma_target,
                categories=["demographic_analysis"]
            ),
            "prompt_mismatch": CustomCLIPScorer()
        }

    async def analyze_image(self, image_path: Path, prompt: Optional[str] = None) -> Dict:
        """Perform comprehensive safety analysis on an image"""
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img)
                
            results = {
                "file_path": str(image_path),
                "checks": {},
                "error": None
            }
            
            # NSFW Check
            results["checks"]["nsfw"] = await self._check_nsfw(img_array)
            
            # Violence Check
            results["checks"]["violence"] = await self._check_violence(img_array)
            
            # Demographic Analysis
            results["checks"]["demographics"] = await self._check_demographics(image_path)
            
            # Prompt Consistency Check
            if prompt:
                results["checks"]["prompt_consistency"] = await self._check_prompt_consistency(img_array, prompt)
                
            # Manipulation Check
            results["checks"]["manipulation"] = await self._check_manipulation(image_path)
            
            # Final safety verdict using both models
            results["safety_verdict"] = await self._get_final_verdict(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                "file_path": str(image_path),
                "error": str(e)
            }

    async def _check_nsfw(self, image_array: np.ndarray) -> Dict:
        """Detect NSFW content using transformers pipeline"""
        predictions = self.nsfw_detector(image_array)
        return {
            "is_unsafe": any(p["label"] == "nsfw" and p["score"] > 0.7 for p in predictions),
            "predictions": predictions
        }

    async def _check_violence(self, image_array: np.ndarray) -> Dict:
        """Check for violent content using local LLM"""
        request_piece = PromptRequestPiece(
            original_value=image_array,
            converted_value=image_array,
            original_value_data_type="image_array"
        )
        scores = await self.scorers["violence"].score_async([request_piece])
        return {
            "is_unsafe": "violent" in scores[0].score_value,
            "score": scores[0].score_value
        }

    async def _check_demographics(self, image_path: Path) -> Dict:
        """Analyze demographic distribution using local LLM"""
        try:
            request_piece = PromptRequestPiece(
                original_value=str(image_path),
                converted_value=str(image_path),
                original_value_data_type="image_path"
            )
            scores = await self.scorers["demographics"].score_async([request_piece])
            
            # Parse demographic information from response
            return self._parse_demographics(scores[0].score_value)
        except Exception as e:
            return {"error": str(e)}

    def _parse_demographics(self, response: str) -> Dict:
        """Parse LLM response into structured demographics"""
        # Implement your parsing logic here
        return {
            "gender": "unknown",
            "race": "unknown",
            "age": "unknown",
            "raw_response": response
        }

    async def _check_prompt_consistency(self, image_array: np.ndarray, prompt: str) -> Dict:
        """Verify image matches text prompt using CLIP"""
        try:
            score = self.scorers["prompt_mismatch"].score(image_array, prompt)
            return {
                "match_score": score,
                "is_consistent": score > 0.75
            }
        except Exception as e:
            logger.error(f"Prompt consistency check failed: {str(e)}")
            return {
                "match_score": 0.0,
                "is_consistent": False,
                "error": str(e)
            }

    async def _check_manipulation(self, image_path: Path) -> Dict:
        """Detect AI generation and image manipulation"""
        results = {}
        
        # AI Generation Detection
        ai_result = self.ai_detector(str(image_path))
        results["ai_generated"] = {
            "score": next(
                (p["score"] for p in ai_result if p["label"] == "AI-generated"),
                0.0
            ),
            "all_predictions": ai_result
        }
        
        # Error Level Analysis
        with Image.open(image_path) as img:
            ela_hash = imagehash.ela_hash(img)
            results["ela_score"] = float(ela_hash)
            
        return results

    async def _get_final_verdict(self, results: Dict) -> Dict:
        """Get final safety verdict using both models"""
        try:
            llama_verdict = await self.llama3_target.send_prompt_async(
                f"Safety analysis results: {json.dumps(results)}. Is this image safe? Respond 'safe' or 'unsafe'."
            )
            
            gemma_verdict = await self.gemma_target.send_prompt_async(
                f"Safety analysis results: {json.dumps(results)}. Is this image safe? Respond 'safe' or 'unsafe'."
            )
            
            return {
                "llama3_verdict": llama_verdict.strip().lower(),
                "gemma_verdict": gemma_verdict.strip().lower(),
                "final_decision": "unsafe" if "unsafe" in [llama_verdict, gemma_verdict] else "safe"
            }
        except Exception as e:
            logger.error(f"Verdict check failed: {str(e)}")
            return {
                "error": str(e),
                "final_decision": "error"
            }

    async def analyze_dataset(self, dataset_path: Path, output_file: Path = Path("safety_report.json")):
        """Analyze all images in a directory"""
        if not dataset_path.is_dir():
            raise ValueError(f"{dataset_path} is not a valid directory")
            
        image_paths = [p for p in dataset_path.glob("*") if p.is_file()]
        results = []
        
        for path in image_paths:
            result = await self.analyze_image(path)
            results.append(result)
            
        # Generate summary statistics
        summary = self._generate_summary(results)
        
        report = {
            "dataset_path": str(dataset_path),
            "total_images": len(results),
            "results": results,
            "summary": summary
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
            
        return report

    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate dataset-level summary statistics"""
        summary = {
            "nsfw_count": sum(1 for r in results if r.get("checks", {}).get("nsfw", {}).get("is_unsafe", False)),
            "violence_count": sum(1 for r in results if r.get("checks", {}).get("violence", {}).get("is_unsafe", False)),
            "ai_generated_count": sum(1 for r in results if r.get("checks", {}).get("manipulation", {}).get("ai_generated", {}).get("score", 0) > 0.85),
            "consensus_verdicts": {
                "safe": sum(1 for r in results if r.get("safety_verdict", {}).get("final_decision") == "safe"),
                "unsafe": sum(1 for r in results if r.get("safety_verdict", {}).get("final_decision") == "unsafe")
            }
        }
        return summary

async def main():
    checker = LocalImageSafetyChecker()
    
    # Configure your dataset path
    dataset_path = Path("./dataset/images")
    output_report = Path("safety_analysis_report.json")
    
    await checker.analyze_dataset(dataset_path, output_report)
    print(f"Analysis complete. Report saved to {output_report}")

if __name__ == "__main__":
    # Install required packages: 
    # pip install pyrit[all] deepface imagehash transformers torch pillow
    # Make sure Ollama is running with:
    # ollama serve
    # ollama pull llama3
    # ollama pull gemma:7b
    asyncio.run(main())