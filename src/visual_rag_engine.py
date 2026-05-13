import torch
import os
import gc
import pickle
from pdf2image import convert_from_path
from PIL import Image
from qwen_vl_utils import process_vision_info



# Use native transformers
from transformers import (
    ColPaliForRetrieval, 
    ColPaliProcessor,
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

class VisualRAGEngine:
    def __init__(self):
        # Initialize attributes
        self.page_images = []
        self.page_embeddings = []
        self.current_doc_name = None
        
        print("👁️ Initializing Visual RAG Engine (ColPali + Qwen2-VL)...")
        
        self.retriever_name = "vidore/colpali-v1.2-hf"
        self.generator_name = "Qwen/Qwen2-VL-2B-Instruct"
        
        # --- DEVICE STRATEGY ---
        if torch.backends.mps.is_available():
            print("🍎 Apple Silicon detected. Applying memory safety fix...")
            self.device_retriever = "cpu"  # Force CPU for stability
            self.device_generator = "mps"  # Use MPS for speed
            self.dtype_retriever = torch.float32
            self.dtype_generator = torch.float16
        elif torch.cuda.is_available():
            self.device_retriever = "cuda"
            self.device_generator = "cuda"
            self.dtype_retriever = torch.bfloat16
            self.dtype_generator = torch.bfloat16
        else:
            self.device_retriever = "cpu"
            self.device_generator = "cpu"
            self.dtype_retriever = torch.float32
            self.dtype_generator = torch.float32
            
        print(f"🚀 Devices -> Retriever: {self.device_retriever}, Generator: {self.device_generator}")

        # --- LOAD RETRIEVER ---
        print("📥 Loading Retriever (ColPali)...")
        try:
            self.retriever = ColPaliForRetrieval.from_pretrained(
                self.retriever_name,
                torch_dtype=self.dtype_retriever,
                device_map=self.device_retriever 
            ).eval()
            self.retriever_processor = ColPaliProcessor.from_pretrained(self.retriever_name)
        except Exception as e:
            print(f"❌ Failed to load retriever: {e}")
            raise

        # --- LOAD GENERATOR ---
        print("🤖 Loading Generator (Qwen2-VL)...")
        try:
            self.generator = Qwen2VLForConditionalGeneration.from_pretrained(
                self.generator_name,
                torch_dtype=self.dtype_generator,
                device_map=self.device_generator,
                trust_remote_code=True
            ).eval()
            self.generator_processor = AutoProcessor.from_pretrained(
                self.generator_name, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Failed to load generator: {e}")
            raise

    def load_document(self, pdf_path, progress_callback=None):
        """
        Loads document with progress updates.
        progress_callback: function(current_step, total_steps, message)
        """
        if not os.path.exists(pdf_path): 
            return False, "File not found."

        self.current_doc_name = os.path.basename(pdf_path)
        
        # Cache paths
        cache_dir = os.path.join(os.path.dirname(pdf_path), ".cache")
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        index_file = os.path.join(cache_dir, f"{self.current_doc_name}.index.pt")
        images_file = os.path.join(cache_dir, f"{self.current_doc_name}.images.pkl")

        # --- PATH A: LOAD FROM DISK ---
        if os.path.exists(index_file) and os.path.exists(images_file):
            print(f"⚡ Found cached index for {self.current_doc_name}.")
            if progress_callback: progress_callback(100, 100, "Loading from cache...")
            try:
                self.page_embeddings = torch.load(index_file, map_location="cpu")
                with open(images_file, "rb") as f:
                    self.page_images = pickle.load(f)
                return True, f"⚡ Loaded {len(self.page_images)} pages from cache instantly."
            except Exception as e:
                print(f"⚠️ Cache corrupted, re-indexing: {e}")

        # --- PATH B: CREATE NEW INDEX ---
        try:
            self.page_images = []
            self.page_embeddings = []
            gc.collect()
            if self.device_generator == "mps": torch.mps.empty_cache()
            
            if progress_callback: progress_callback(0, 100, "Converting PDF to images...")
            print(f"📄 Converting {pdf_path} to images...")
            images = convert_from_path(pdf_path, dpi=100) 
            self.page_images = images
            
            total_pages = len(images)
            print(f"🧠 Encoding {total_pages} pages...")
            
            with torch.no_grad():
                for idx, img in enumerate(images):
                    # Update Progress
                    if progress_callback:
                        msg = f"Encoding Page {idx+1}/{total_pages}"
                        progress_callback(idx+1, total_pages, msg)

                    # Process on CPU
                    batch = self.retriever_processor.process_images([img]).to(self.device_retriever)
                    outputs = self.retriever(**batch)
                    
                    if hasattr(outputs, "last_hidden_state"): emb = outputs.last_hidden_state
                    elif isinstance(outputs, torch.Tensor): emb = outputs
                    else: emb = outputs[0]
                    
                    self.page_embeddings.append(emb.cpu())
            
            # Save
            if progress_callback: progress_callback(100, 100, "Saving to disk...")
            print("💾 Saving index to disk...")
            torch.save(self.page_embeddings, index_file)
            with open(images_file, "wb") as f:
                pickle.dump(self.page_images, f)
            
            return True, f"Indexed & Saved {total_pages} pages."
        
        except Exception as e:
            return False, str(e)

    def delete_document(self, pdf_path):
        """Deletes the PDF and its cached index/image files."""
        if not os.path.exists(pdf_path): return False, "File not found."
            
        try:
            doc_name = os.path.basename(pdf_path)
            cache_dir = os.path.join(os.path.dirname(pdf_path), ".cache")
            
            files_to_remove = [
                pdf_path,
                os.path.join(cache_dir, f"{doc_name}.index.pt"),
                os.path.join(cache_dir, f"{doc_name}.images.pkl")
            ]
            
            for fpath in files_to_remove:
                if os.path.exists(fpath): os.remove(fpath)
            
            if self.current_doc_name == doc_name:
                self.page_images = []
                self.page_embeddings = []
                self.current_doc_name = None
                
            return True, f"Deleted {doc_name}."
        except Exception as e:
            return False, f"Error: {e}"

    def query(self, user_query, top_k=1):
        if not self.page_embeddings: return None, "No document loaded."
        print(f"🔎 Searching for: '{user_query}'")
        
        # --- STEP 1: RETRIEVAL (CPU) ---
        with torch.no_grad():
            query_batch = self.retriever_processor.process_queries([user_query]).to(self.device_retriever)
            query_outputs = self.retriever(**query_batch)
            
            if isinstance(query_outputs, torch.Tensor):
                q_emb = query_outputs
            else:
                q_emb = query_outputs[0]

            scores = []
            for idx, page_emb in enumerate(self.page_embeddings):
                if isinstance(page_emb, torch.Tensor): p_emb = page_emb
                else: p_emb = page_emb[0]
                
                p_emb_safe = p_emb.to(self.device_retriever)
                
                try:
                    score_tensor = self.retriever_processor.score_retrieval(q_emb, p_emb_safe)
                    score_val = score_tensor.item()
                except:
                    # Fallback (flatten cosine)
                    score_val = torch.nn.functional.cosine_similarity(
                        q_emb.view(1, -1), p_emb_safe.view(1, -1), dim=1
                    ).mean().item()
                
                scores.append((score_val, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_idx = scores[0][1]
        best_page_img = self.page_images[best_idx]
        
        # --- STEP 2: GENERATION (MPS) ---
        print(f"📝 Generating answer from Page {best_idx + 1}...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": best_page_img},
                    {"type": "text", "text": f"Answer based ONLY on the image: {user_query}"},
                ],
            }
        ]
        
        text = self.generator_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.generator_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device_generator)

        with torch.no_grad():
            generated_ids = self.generator.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.generator_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return {
            "answer": output_text.strip(),
            "image": best_page_img,
            "page": best_idx + 1,
            "score": round(scores[0][0], 4)
        }, "Success"
