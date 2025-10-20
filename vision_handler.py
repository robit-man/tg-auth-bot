#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_handler.py — Image analysis using Ollama vision models

Handles:
- Image uploads via Telegram
- Vision analysis using Ollama multimodal models
- Base64 encoding for API calls
- Context integration for image-aware conversations
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None  # type: ignore


@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    success: bool
    description: str
    error: Optional[str] = None
    model_used: str = ""
    image_path: str = ""


class VisionHandler:
    """
    Handler for image analysis using Ollama vision models.

    Based on examples/context.py vision extraction and examples/tools.py describe_image
    """

    def __init__(
        self,
        vision_model: str = "llava",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize vision handler.

        Args:
            vision_model: Ollama vision model name (llava, bakllava, etc.)
            ollama_base_url: Ollama API endpoint
        """
        self.vision_model = vision_model
        self.ollama_base_url = ollama_base_url

    def analyze_image(
        self,
        image_path: Path,
        *,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> ImageAnalysisResult:
        """
        Analyze an image using the vision model.

        Args:
            image_path: Path to the image file
            user_prompt: Optional custom prompt (default: "Describe this image in detail")
            system_prompt: Optional system prompt
            temperature: Model temperature (0.0 = deterministic)
            stream: Whether to stream the response

        Returns:
            ImageAnalysisResult with description
        """
        if not OLLAMA_AVAILABLE:
            return ImageAnalysisResult(
                success=False,
                description="",
                error="Ollama not available - install with: pip install ollama",
            )

        if not image_path.exists():
            return ImageAnalysisResult(
                success=False,
                description="",
                error=f"Image not found: {image_path}",
            )

        # Default prompts
        if system_prompt is None:
            system_prompt = "You are a helpful visual assistant. Describe images accurately and in detail."

        if user_prompt is None:
            user_prompt = "Describe this image in detail. What do you see?"

        try:
            # Read and encode image
            image_bytes = image_path.read_bytes()
            b64_image = base64.b64encode(image_bytes).decode("ascii")

            # Build messages
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt,
                })

            messages.append({
                "role": "user",
                "content": user_prompt,
                "images": [b64_image],
            })

            # Call Ollama
            if stream:
                description = self._stream_response(messages, temperature)
            else:
                response = ollama.chat(
                    model=self.vision_model,
                    messages=messages,
                    options={"temperature": temperature},
                )
                description = response["message"]["content"]

            return ImageAnalysisResult(
                success=True,
                description=description.strip(),
                model_used=self.vision_model,
                image_path=str(image_path),
            )

        except Exception as e:
            return ImageAnalysisResult(
                success=False,
                description="",
                error=f"Vision analysis failed: {str(e)}",
                model_used=self.vision_model,
                image_path=str(image_path),
            )

    def _stream_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
    ) -> str:
        """Stream response from vision model"""
        pieces: List[str] = []

        try:
            for chunk in ollama.chat(
                model=self.vision_model,
                messages=messages,
                stream=True,
                options={"temperature": temperature},
            ):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    pieces.append(delta)

            print()  # Newline after streaming
            return "".join(pieces)

        except Exception:
            return "".join(pieces) if pieces else ""

    def analyze_with_context(
        self,
        image_path: Path,
        conversation_context: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
    ) -> ImageAnalysisResult:
        """
        Analyze image with conversation context.

        This allows the vision model to answer questions about the image
        in the context of an ongoing conversation.

        Args:
            image_path: Path to image
            conversation_context: Previous messages [{"role": "user"|"assistant", "content": "..."}]
            temperature: Model temperature

        Returns:
            ImageAnalysisResult
        """
        if not OLLAMA_AVAILABLE:
            return ImageAnalysisResult(
                success=False,
                description="",
                error="Ollama not available",
            )

        try:
            # Encode image
            b64_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

            # Build messages with context
            messages = []

            # Add conversation context
            for msg in conversation_context:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            # Add image to the last user message
            if messages and messages[-1]["role"] == "user":
                messages[-1]["images"] = [b64_image]
            else:
                # If last message wasn't from user, create a new one
                messages.append({
                    "role": "user",
                    "content": "Please analyze this image in the context of our conversation.",
                    "images": [b64_image],
                })

            # Call Ollama
            response = ollama.chat(
                model=self.vision_model,
                messages=messages,
                options={"temperature": temperature},
            )

            description = response["message"]["content"]

            return ImageAnalysisResult(
                success=True,
                description=description.strip(),
                model_used=self.vision_model,
                image_path=str(image_path),
            )

        except Exception as e:
            return ImageAnalysisResult(
                success=False,
                description="",
                error=f"Context-aware analysis failed: {str(e)}",
            )

    def create_image_context_for_chat(
        self,
        image_path: Path,
        user_message: str,
    ) -> Dict[str, Any]:
        """
        Create a message dict for Ollama chat API with image.

        This can be directly added to the messages array for a chat call.

        Args:
            image_path: Path to image
            user_message: User's message about the image

        Returns:
            Message dict with embedded image
        """
        b64_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

        return {
            "role": "user",
            "content": user_message,
            "images": [b64_image],
        }

    def describe_multiple_images(
        self,
        image_paths: List[Path],
        *,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> List[ImageAnalysisResult]:
        """
        Analyze multiple images.

        Args:
            image_paths: List of image paths
            prompt: Optional custom prompt for each image
            temperature: Model temperature

        Returns:
            List of ImageAnalysisResult objects
        """
        results = []

        for image_path in image_paths:
            result = self.analyze_image(
                image_path,
                user_prompt=prompt,
                temperature=temperature,
            )
            results.append(result)

        return results

    def compare_images(
        self,
        image1_path: Path,
        image2_path: Path,
        *,
        comparison_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ImageAnalysisResult:
        """
        Compare two images.

        Args:
            image1_path: First image
            image2_path: Second image
            comparison_prompt: Custom comparison prompt
            temperature: Model temperature

        Returns:
            ImageAnalysisResult with comparison
        """
        if not OLLAMA_AVAILABLE:
            return ImageAnalysisResult(
                success=False,
                description="",
                error="Ollama not available",
            )

        if comparison_prompt is None:
            comparison_prompt = "Compare these two images. What are the similarities and differences?"

        try:
            # Encode both images
            b64_img1 = base64.b64encode(image1_path.read_bytes()).decode("ascii")
            b64_img2 = base64.b64encode(image2_path.read_bytes()).decode("ascii")

            # Build message with both images
            messages = [
                {
                    "role": "user",
                    "content": comparison_prompt,
                    "images": [b64_img1, b64_img2],
                }
            ]

            # Call Ollama
            response = ollama.chat(
                model=self.vision_model,
                messages=messages,
                options={"temperature": temperature},
            )

            description = response["message"]["content"]

            return ImageAnalysisResult(
                success=True,
                description=description.strip(),
                model_used=self.vision_model,
                image_path=f"{image1_path} vs {image2_path}",
            )

        except Exception as e:
            return ImageAnalysisResult(
                success=False,
                description="",
                error=f"Image comparison failed: {str(e)}",
            )


def create_vision_handler(model: str = "llava") -> VisionHandler:
    """
    Factory function to create a VisionHandler.

    Args:
        model: Ollama vision model name

    Returns:
        Configured VisionHandler instance
    """
    return VisionHandler(vision_model=model)
