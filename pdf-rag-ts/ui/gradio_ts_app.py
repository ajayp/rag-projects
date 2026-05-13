import os
import gradio as gr
import requests

TS_SERVER_URL = "http://localhost:3000"

_path_cache: dict = {}
_strategy_cache: dict = {}


def _refresh_ui_state():
    global _strategy_cache
    try:
        response = requests.get(f"{TS_SERVER_URL}/stats")
        if response.status_code != 200:
            return "*TS Server not reachable.*", [], {}, {}
        stats = response.json()
    except Exception:
        return "*TS Server not reachable.*", [], {}, {}

    path_map = {os.path.basename(p): p for p in stats.get("documents", {})}
    strategy_map = {
        os.path.basename(p): info.get("strategy", "")
        for p, info in stats.get("documents", {}).items()
    }

    if not stats.get("documents"):
        return "*No documents indexed yet.*", [], path_map, strategy_map

    total_chunks = stats.get('totalChunks', 0)
    lines = [f"**{total_chunks} chunks** across {len(stats['documents'])} document(s)\n"]
    for path, info in stats["documents"].items():
        name = os.path.basename(path)
        strategy = info.get("strategy", "")
        strategy_label = f" · `{strategy}`" if strategy else ""
        lines.append(f"- {name}: {info['chunks']} chunks{strategy_label}")

    return "\n".join(lines), list(path_map.keys()), path_map, strategy_map


def process_pdf(file, strategy):
    global _path_cache, _strategy_cache
    if file is None:
        return gr.update(), gr.update(), gr.update(), ""
    basename = os.path.basename(file.name)
    if basename in _path_cache:
        return f"❌ '{basename}' is already indexed. Press Reset to clear the collection first.", gr.update(), gr.update(), ""
    try:
        response = requests.post(f"{TS_SERVER_URL}/process", json={
            "filePath": file.name,
            "strategy": strategy,
            "useLlamaParse": True
        })
        if response.status_code == 200:
            stats_md, choices, _path_cache, _strategy_cache = _refresh_ui_state()
            return "✅ Document processed (TS).", stats_md, gr.update(choices=choices, value=choices[-1] if choices else None), ""
        elif response.status_code == 409:
            return f"❌ {response.json().get('error', 'Already indexed.')}", gr.update(), gr.update(), ""
        return f"❌ Error: {response.text}", gr.update(), gr.update(), ""
    except Exception as e:
        return f"❌ Connection error: {e}", gr.update(), gr.update(), ""


def reset_collection():
    global _path_cache, _strategy_cache
    requests.post(f"{TS_SERVER_URL}/reset")
    _path_cache = {}
    _strategy_cache = {}
    return "✅ Collection reset.", "*No documents indexed yet.*", gr.update(choices=[], value=None), ""


def show_strategy(doc_choice):
    if not doc_choice:
        return ""
    s = _strategy_cache.get(doc_choice, "")
    return f"*Strategy: {s}*" if s else ""


def submit(message, history, doc_choice, rewrite, alpha, use_hyde):
    if not message.strip():
        return history, ""

    response = requests.post(f"{TS_SERVER_URL}/query", json={
        "question": message,
        "options": {
            "sourceFile": doc_choice,
            "alpha": alpha,
            "useHyDE": use_hyde,
            "useRewrite": rewrite
        }
    })

    answer = response.json().get("answer", "Error communicating with TS server.")
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


_initial_stats_md, _initial_choices, _path_cache, _strategy_cache = _refresh_ui_state()
_initial_strategy = show_strategy(_initial_choices[0] if _initial_choices else None)

with gr.Blocks(title="TS RAG Tester") as demo:
    gr.Markdown("# 🤖 TS RAG — Multi-Strategy Tester")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Documents")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            strategy_dropdown = gr.Dropdown(
                label="Chunking Strategy",
                choices=["STANDARD", "HIERARCHICAL", "CONTEXTUAL", "FULL_PAGE"],
                value="STANDARD"
            )
            process_btn = gr.Button("Process (via TS)", variant="primary")
            reset_btn = gr.Button("Reset", variant="stop")
            process_status = gr.Textbox(interactive=False, show_label=False)
            stats_md = gr.Markdown(_initial_stats_md)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500)
            doc_dropdown = gr.Dropdown(
                label="Document Filter",
                choices=_initial_choices,
                value=_initial_choices[0] if _initial_choices else None,
            )
            strategy_display = gr.Markdown(_initial_strategy)
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Ask a question…", scale=5, show_label=False)
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            rewrite_toggle = gr.Checkbox(label="Query expansion", value=False, info="Adds synonyms and related terms before querying — helps when your wording differs from the document's")
            hyde_toggle = gr.Checkbox(label="HyDE", value=False, info="Generates a hypothetical answer and searches with that — helps bridge the gap between short questions and long document passages")
            nli_toggle = gr.Checkbox(label="NLI (contradiction detection) - coming soon", value=False, interactive=False, info="Checks generated answer against source chunks for contradictions")
            multi_hop = gr.Checkbox(label="Multi-Hop retrieval - coming soon", value=False, interactive=False, info="Retrieval is sequential evidence gathering across multiple dependent retrieval steps.")
            semantic_match_toggle = gr.Checkbox(label="Semantic matching - coming soon", value=False, interactive=False, info="Re-ranks results using semantic similarity scoring")
            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="Search mode", info="0 = keyword only (BM25) · 1 = semantic only (vector)")

    process_btn.click(process_pdf, [pdf_input, strategy_dropdown], [process_status, stats_md, doc_dropdown, strategy_display])
    reset_btn.click(reset_collection, outputs=[process_status, stats_md, doc_dropdown, strategy_display])
    doc_dropdown.change(show_strategy, [doc_dropdown], [strategy_display])
    submit_btn.click(submit, [msg_input, chatbot, doc_dropdown, rewrite_toggle, alpha_slider, hyde_toggle], [chatbot, msg_input])
    msg_input.submit(submit, [msg_input, chatbot, doc_dropdown, rewrite_toggle, alpha_slider, hyde_toggle], [chatbot, msg_input])

if __name__ == "__main__":
    demo.launch(server_port=7861)
