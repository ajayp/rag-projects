import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import gradio as gr
from src.rag_system import LocalRAGSystem
from src.caching.semantic_cache import SemanticCache

_cache = SemanticCache()
if _cache.is_available():
    print("✅ Semantic cache connected (Redis)")
else:
    print("⚠️  Redis not reachable — semantic cache disabled")
    _cache = None

# Initialized at module level because Gradio wires event handlers at import time.
# Ensure Weaviate (port 8080) and Ollama (port 11434) are running before launching.
try:
    rag = LocalRAGSystem(
        embedding_model="nomic-embed-text",
        generative_model="qwen2.5:14b",
        cache=_cache,
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize LocalRAGSystem: {e}\n"
        "Make sure Weaviate (docker compose up -d) and Ollama are running."
    ) from e

# basename → full path; rebuilt after each index/reset to avoid per-query chunk fetches
_path_cache: dict[str, str] = {}


def _refresh_ui_state() -> tuple[str, list, dict]:
    """Returns (stats_markdown, doc_choices, updated _path_cache) from one stats fetch."""
    stats = rag.get_document_stats()
    path_map = {os.path.basename(p): p for p in stats["documents"]}

    if not stats["documents"]:
        return "*No documents indexed yet.*", [], path_map

    lines = [f"**{stats['total_chunks']} chunks** across {len(stats['documents'])} document(s)\n"]
    for path, info in stats["documents"].items():
        name = os.path.basename(path)
        pages = info.get("pages", "?")
        lines.append(f"- {name}: {info['chunks']} chunks, {pages} pages")

    return "\n".join(lines), list(path_map.keys()), path_map


def resolve_source_file(basename: str):
    if not basename:
        return None
    return _path_cache.get(basename)


def process_pdf(file):
    global _path_cache
    if file is None:
        return gr.update(), gr.update(), gr.update()
    rag.process_document(file.name, use_llamaparse=True)
    stats_md, choices, _path_cache = _refresh_ui_state()
    return "✅ Document processed and indexed.", stats_md, gr.update(choices=choices, value=choices[-1] if choices else None)


def reset_collection():
    global _path_cache
    rag.reset()
    _path_cache = {}
    return "✅ Collection reset.", "*No documents indexed yet.*", gr.update(choices=[], value=None)


def submit(message, history, doc_choice, rewrite, alpha, use_hyde, use_cache):
    if not message.strip():
        return history, ""
    source_file = resolve_source_file(doc_choice)
    answer = rag.ask_question(
        message,
        source_file=source_file,
        alpha=alpha,
        use_hyde=use_hyde,
        use_rewrite=rewrite,
        use_cache=use_cache,
    )
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


# Seed path cache and initial UI state on startup
_initial_stats_md, _initial_choices, _path_cache = _refresh_ui_state()

with gr.Blocks(title="Local RAG") as demo:
    gr.Markdown("# 🤖 Local RAG — Page-Aware Document Q&A")

    with gr.Row():
        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### 📄 Documents")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document", variant="primary")
            reset_btn = gr.Button("Reset Collection", variant="stop")
            process_status = gr.Textbox(interactive=False, lines=1, show_label=False)
            gr.Markdown("### 📊 Indexed")
            stats_md = gr.Markdown(_initial_stats_md)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500, buttons=["copy_all"])
            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Document",
                    choices=_initial_choices,
                    value=_initial_choices[0] if _initial_choices else None,
                )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question…",
                    scale=5,
                    show_label=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            rewrite_toggle = gr.Checkbox(label="Query expansion", value=False, info="Adds synonyms and related terms before querying — helps when your wording differs from the document's")
            hyde_toggle = gr.Checkbox(label="HyDE", value=False, info="Generates a hypothetical answer and searches with that — helps bridge the gap between short questions and long document passages")
            cache_toggle = gr.Checkbox(label="Semantic cache", value=True, info="Serve repeated or similar questions from Redis instead of hitting the LLM again")
            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="Search mode", info="0 = keyword only (BM25) · 1 = semantic only (vector)")

    process_btn.click(
        process_pdf,
        inputs=[pdf_input],
        outputs=[process_status, stats_md, doc_dropdown],
    )
    reset_btn.click(
        reset_collection,
        inputs=[],
        outputs=[process_status, stats_md, doc_dropdown],
    )
    submit_btn.click(submit, [msg_input, chatbot, doc_dropdown, rewrite_toggle, alpha_slider, hyde_toggle, cache_toggle], [chatbot, msg_input])
    msg_input.submit(submit, [msg_input, chatbot, doc_dropdown, rewrite_toggle, alpha_slider, hyde_toggle, cache_toggle], [chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
