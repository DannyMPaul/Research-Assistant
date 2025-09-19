import os
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class LLMService:
    def __init__(self):
        self.enabled = LLM_AVAILABLE and self._check_hardware()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Smaller fallback model
        
        if self.enabled:
            try:
                self._initialize_model()
            except Exception as e:
                print(f"LLM initialization failed: {e}")
                self.enabled = False
    
    def _check_hardware(self) -> bool:
        if not torch.cuda.is_available():
            return False
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_memory >= 8.0  # Minimum 8GB GPU memory
    
    def _initialize_model(self):
        if not self.enabled:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                load_in_8bit=True if torch.cuda.is_available() else False
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map,
                torch_dtype=torch_dtype,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.enabled = False
    
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
        if not self.enabled:
            return {
                "answer": "AI Q&A service is currently unavailable. Please use the search function instead.",
                "sources": [],
                "confidence": 0.0,
                "error": "LLM service disabled"
            }
        
        try:
            prompt = self.build_context_prompt(question, context_chunks, conversation_history)
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._generate_sync, prompt)
            
            answer = self._extract_answer(response)
            sources = self._extract_sources(context_chunks)
            confidence = self._calculate_confidence(answer, context_chunks)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "tokens_used": len(self.tokenizer.encode(prompt + answer)) if answer else 0
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
            print(f"Generation error: {e}")
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