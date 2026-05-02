import os
import gradio as gr
from cli import LocalRAGSystem

rag = LocalRAGSystem(
    embedding_model="nomic-embed-text",
    generative_model="qwen2.5:14b",
)

def get_stats_md() -> str:
    stats = rag.get_document_stats()
    if not stats["documents"]:
        return "*No documents indexed yet.*"
    lines = [f"**{stats['total_chunks']} chunks** across {len(stats['documents'])} document(s)\n"]
    for path, info in stats["documents"].items():
        name = os.path.basename(path)
        pages = info.get("pages", "?")
        lines.append(f"- {name}: {info['chunks']} chunks, {pages} pages")
    return "\n".join(lines)


def get_doc_choices() -> list:
    stats = rag.get_document_stats()
    return [os.path.basename(p) for p in stats["documents"]]


def resolve_source_file(basename: str):
    if not basename:
        return None
    stats = rag.get_document_stats()
    for path in stats["documents"]:
        if os.path.basename(path) == basename:
            return path
    return None


def process_pdf(file):
    if file is None:
        return gr.update(), get_stats_md(), gr.update()
    rag.process_document(file.name, use_llamaparse=True)
    choices = get_doc_choices()
    return "✅ Document processed and indexed.", get_stats_md(), gr.update(choices=choices, value=choices[-1] if choices else None)


def reset_collection():
    rag.reset()
    return "✅ Collection reset.", get_stats_md(), gr.update(choices=[], value=None)


def submit(message, history, doc_choice, rewrite, alpha, use_hyde):
    if not message.strip():
        return history, ""
    query = rag.rewrite_query(message) if rewrite else message
    source_file = resolve_source_file(doc_choice)
    answer = rag.ask_question(query, source_file=source_file, alpha=alpha, use_hyde=use_hyde)
    if rewrite and query != message:
        answer = f"🔍 *Expanded query: \"{query}\"*\n\n{answer}"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


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
            stats_md = gr.Markdown(get_stats_md())

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500, buttons=["copy_all"])
            with gr.Row():
                _initial_choices = get_doc_choices()
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
    submit_btn.click(submit, [msg_input, chatbot, doc_dropdown, rewrite_toggle, alpha_slider, hyde_toggle], [chatbot, msg_input])
    msg_input.submit(submit, [msg_input, chatbot, doc_dropdown, rewrite_toggle, alpha_slider, hyde_toggle], [chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
