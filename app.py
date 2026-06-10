"""
app.py — Gradio Web Interface for Professor Reviews RAG (Milestone 5)
Wraps query.py's ask() function in a shareable web UI.

Run: python app.py
     → Opens local URL + prints a public share URL for your demo video.
"""

import gradio as gr
from query import ask   # imports model + ChromaDB on first load

# ──────────────────────────────────────────────
# Handler
# ──────────────────────────────────────────────

def handle_query(question: str):
    """
    Called by Gradio on every submission.
    Returns (answer_markdown, sources_text) for the two output components.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    result = ask(question)

    # Format answer as markdown
    answer_md = result["answer"]

    # Format sources as a bulleted list
    if result["sources"]:
        sources_text = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources_text = "No sources retrieved."

    return answer_md, sources_text


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="The Unofficial Guide: Professor Reviews (UCR)") as demo:

    gr.Markdown(
        """
        # 📚 The Unofficial Guide: Professor Reviews (UCR)
        Ask questions about professors based on real **Rate My Professors** reviews.
        Answers are grounded strictly in student reviews — no outside knowledge used.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Is Thomas Kuhlman's physics class curved?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            sources_box = gr.Textbox(
                label="📂 Sources Used",
                lines=6,
                interactive=False,
            )

    answer_box = gr.Markdown(label="Answer")

    # Example questions from the evaluation plan
    gr.Examples(
        examples=[
            ["What do students say about Stefano Lonardi's exam grading?"],
            ["Does Derek Mkhaiel give good feedback on assignments?"],
            ["Is Thomas Kuhlman's physics class curved?"],
            ["Does Marko Spasojevic offer extra credit in BIOL003?"],
            ["What do students say about Matthew Lang's teaching style?"],
            ["Who won the Super Bowl?"],   # out-of-scope grounding test
        ],
        inputs=question_box,
    )

    gr.Markdown(
        """
        ---
        *Powered by [Rate My Professors](https://www.ratemyprofessors.com) reviews · 
        Retrieval: ChromaDB + all-MiniLM-L6-v2 · 
        Generation: Groq llama-3.3-70b-versatile*
        """
    )

    # Wire up button and Enter key
    submit_btn.click(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )
    question_box.submit(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )

# ──────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[app.py] Starting Gradio interface...")
    demo.launch(
        share=True,          # prints a public gradio.live URL for demo video
        show_error=True,     # surface Python errors in the browser
    )