import streamlit as st
import os
from src.visual_rag_engine import VisualRAGEngine
from src.config import DOCS_DIR

st.set_page_config(page_title="ACC-RAG Visual Expert", layout="wide", page_icon="👁️")

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    .stImage { border-radius: 8px; border: 1px solid #ddd; }
    .stSpinner { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Initialize Engine
if "engine" not in st.session_state:
    with st.spinner("🚀 Booting Visual AI Engine..."):
        try:
            st.session_state.engine = VisualRAGEngine()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Document Library")
    
    # Upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded_file:
        if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")

    st.divider()

    # File List
    if os.path.exists(DOCS_DIR):
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    else:
        files = []
        
    selected_file = st.selectbox("Select Active Document:", files if files else ["No PDFs found"])
    
    # Buttons Layout
    col1, col2 = st.columns([2, 1])
    
        # ... inside st.sidebar ...
    
    with col1:
        # Load Button
        if st.button("Load & Index", type="primary", disabled=not files, use_container_width=True):
            if selected_file and selected_file != "No PDFs found":
                full_path = os.path.join(DOCS_DIR, selected_file)
                
                # Create progress bar containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, message):
                    percent = int((current / total) * 100)
                    progress_bar.progress(percent)
                    status_text.text(f"⏳ {message}")

                # Call load_document with the callback
                success, msg = st.session_state.engine.load_document(
                    full_path, 
                    progress_callback=update_progress
                )
                
                # Cleanup UI after done
                progress_bar.empty()
                status_text.empty()
                
                if success:
                    st.success(msg)
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error(msg)

    
    with col2:
        # Delete Button
        if st.button("🗑️", help="Delete Document", disabled=not files, use_container_width=True):
             if selected_file and selected_file != "No PDFs found":
                full_path = os.path.join(DOCS_DIR, selected_file)
                success, msg = st.session_state.engine.delete_document(full_path)
                if success:
                    st.toast(msg, icon="🗑️")
                    if st.session_state.engine.current_doc_name is None:
                         st.session_state.messages = []
                    st.rerun()
                else:
                    st.error(msg)

    st.info("""
    ℹ️ **Visual RAG Mode**
    1. **Retrieves** page visually (ColPali).
    2. **Reads** page with Vision AI (Qwen2-VL).
    """)

# --- Chat Interface ---
st.title("👁️ ACC-RAG Visual Expert")

if not st.session_state.engine.current_doc_name:
    st.info("👈 Please load a document to begin.")
else:
    st.caption(f"📖 Analyzing: **{st.session_state.engine.current_doc_name}**")

# History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            with st.expander(f"📄 Source: Page {msg['images'][0]['page']}"):
                st.image(msg["images"][0]["image"], caption=f"Page {msg['images'][0]['page']}")

# Input
if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("👀 Thinking..."):
            result, status = st.session_state.engine.query(prompt)
            
            if result:
                st.markdown(result['answer'])
                with st.expander(f"📄 Source: Page {result['page']}", expanded=True):
                    st.image(result['image'], caption=f"Page {result['page']} • Score: {result['score']:.2f}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "images": [{"image": result['image'], "page": result['page'], "score": result['score']}]
                })
            else:
                st.error("No answer found.")
