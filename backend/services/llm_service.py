import os
import logging
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
except Exception as e:
    LLM_AVAILABLE = False
    logger.warning(f"Transformers import error: {e}")

class LLMService:
    def __init__(self):
        self.enabled = True
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        try:
            self._initialize_model()
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
    
    def _check_hardware(self) -> bool:
        if not LLM_AVAILABLE:
            return False
        if not torch.cuda.is_available():
            return False
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_memory >= 8.0  # Minimum 8GB GPU memory
    
    def _initialize_model(self):
        if not LLM_AVAILABLE:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype if torch_dtype else None,
                device_map=device_map if device_map else None,
                trust_remote_code=True,
                load_in_8bit=True if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else False
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map if device_map else None,
                torch_dtype=torch_dtype if torch_dtype else None,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
    
    def build_context_prompt(self, question: str, context_chunks: List[Dict], 
                           conversation_history: List[Dict] = None) -> str:
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for exchange in conversation_history[-3:]:
                prompt_parts.append(f"User: {exchange.get('question', '')}")
                prompt_parts.append(f"Assistant: {exchange.get('answer', '')}")
            prompt_parts.append("")
        
        prompt_parts.append("Document context:")
        for i, chunk in enumerate(context_chunks[:5], 1):
            filename = chunk.get('filename', 'Unknown')
            content = chunk.get('chunk_text', '')[:500]
            prompt_parts.append(f"Source {i} ({filename}): {content}")
        
        prompt_parts.append("")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("")
        prompt_parts.append("Based on the document context above, provide a helpful and accurate answer. "
                          "If the context doesn't contain enough information, say so clearly. "
                          "Keep your answer concise and cite specific sources when possible.")
        
        return "\n".join(prompt_parts)
    
    async def generate_answer(self, question: str, context_chunks: List[Dict], 
                            conversation_history: List[Dict] = None) -> Dict:
        # If real model exists, use it; else fallback to heuristic answer
        
        try:
            if not question:
                return {
                    "answer": "Please provide a question.",
                    "sources": [],
                    "confidence": 0.0,
                    "tokens_used": 0
                }

            prompt = self.build_context_prompt(question, context_chunks, conversation_history)
            
            if self.pipeline:
                try:
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(None, self._generate_sync, prompt)
                    answer = self._extract_answer(response) if response else self._heuristic_answer(question, context_chunks)
                except Exception as e:
                    logger.error(f"Pipeline generation error: {e}")
                    answer = self._heuristic_answer(question, context_chunks)
            else:
                answer = self._heuristic_answer(question, context_chunks)
                
            sources = self._extract_sources(context_chunks)
            confidence = self._calculate_confidence(answer, context_chunks)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "tokens_used": len(self.tokenizer.encode(prompt + (answer or ""))) if self.tokenizer and answer else 0
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _generate_sync(self, prompt: str) -> str:
        if not self.pipeline:
            return ""
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]["generated_text"]
            return generated_text[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def _extract_answer(self, response: str) -> str:
        if not response:
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        # Clean up the response
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('User:') and not line.startswith('Assistant:'):
                cleaned_lines.append(line)
        
        answer = ' '.join(cleaned_lines)
        
        # Truncate if too long
        if len(answer) > 1000:
            sentences = answer.split('.')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 1000:
                    break
                truncated.append(sentence)
                current_length += len(sentence)
            
            answer = '.'.join(truncated).strip()
            if answer and not answer.endswith('.'):
                answer += '.'
        
        return answer if answer else "I couldn't find a clear answer in the provided context."

    def _heuristic_answer(self, question: str, context_chunks: List[Dict]) -> str:
        # Simple extractive heuristic: pick the highest similarity chunk text
        if not context_chunks:
            return "I couldn't find relevant context to answer that. Try embedding documents and ask again."
        best = max(context_chunks, key=lambda c: c.get('similarity_score', 0.0))
        snippet = best.get('chunk_text', '')
        return f"From the most relevant source: {snippet}"
    
    def _extract_sources(self, context_chunks: List[Dict]) -> List[Dict]:
        sources = []
        for chunk in context_chunks[:5]:
            sources.append({
                "filename": chunk.get('filename', 'Unknown'),
                "chunk_id": chunk.get('chunk_id', 0),
                "similarity_score": round(chunk.get('similarity_score', 0.0), 3),
                "preview": chunk.get('chunk_text', '')[:200] + "..." if len(chunk.get('chunk_text', '')) > 200 else chunk.get('chunk_text', '')
            })
        return sources
    
    def _calculate_confidence(self, answer: str, context_chunks: List[Dict]) -> float:
        if not answer or not context_chunks:
            return 0.0
        
        # Simple confidence calculation based on context quality
        avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks[:3]) / min(3, len(context_chunks))
        
        # Adjust based on answer length and quality indicators
        length_factor = min(1.0, len(answer) / 100)  # Longer answers might be more confident
        
        # Check for uncertainty phrases
        uncertainty_phrases = ["i don't know", "unclear", "not sure", "couldn't find", "not enough information"]
        uncertainty_penalty = 0.3 if any(phrase in answer.lower() for phrase in uncertainty_phrases) else 0.0
        
        confidence = (avg_similarity * 0.7 + length_factor * 0.3) - uncertainty_penalty
        return max(0.0, min(1.0, confidence))

# Global instance
llm_service = LLMService()