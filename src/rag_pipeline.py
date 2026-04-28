from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_gateway import generate_text

DEFAULT_EMBEDDER = "all-mpnet-base-v2"


def load_embedder(model_name: str = DEFAULT_EMBEDDER) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def retrieve_similar_tickets(
    ticket_text: str,
    tickets: pd.DataFrame,
    embedder: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    if "Embedding" not in tickets.columns:
        raise ValueError("The clustered tickets data does not contain an 'Embedding' column.")

    valid_tickets = tickets[tickets["Embedding"].notna()].copy()
    if valid_tickets.empty:
        raise ValueError("No ticket embeddings are available for similarity search.")

    new_embedding = embedder.encode([ticket_text])
    embeddings = np.vstack(valid_tickets["Embedding"].values)
    similarities = cosine_similarity(new_embedding, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    similar_tickets = valid_tickets.iloc[top_indices].copy()
    similar_tickets["Similarity"] = similarities[top_indices]
    return similar_tickets


def build_resolution_prompt(ticket_text: str, similar_tickets: pd.DataFrame) -> str:
    context_lines = []
    for index, (_, row) in enumerate(similar_tickets.iterrows(), start=1):
        context_lines.append(
            "\n".join(
                [
                    f"--- Ticket {index} (Similarity: {row['Similarity']:.2f}) ---",
                    f"Description: {row.get('Short_Description', '')} {row.get('Description_Text', '')}".strip(),
                    f"Resolution Notes: {row.get('Resolution_Notes', '')}",
                ]
            )
        )

    context = "\n\n".join(context_lines)
    return f"""You are an expert IT support agent.

A new ticket has been submitted:
"{ticket_text}"

Here are the most similar historical tickets and how they were resolved:
{context}

Based ONLY on the historical resolution notes, provide a concise, step-by-step recommended solution for the new ticket.
If the historical tickets do not contain a clear solution, state that human review is needed.
"""


def generate_resolution_recommendation(
    ticket_text: str,
    similar_tickets: pd.DataFrame,
    model: str | None = None,
) -> str:
    prompt = build_resolution_prompt(ticket_text, similar_tickets)
    return generate_text(
        prompt,
        model=model,
        max_output_tokens=500,
    )


def resolve_ticket(
    ticket_text: str,
    tickets: pd.DataFrame,
    embedder: SentenceTransformer,
    top_k: int = 5,
    model: str | None = None,
) -> Tuple[str, pd.DataFrame]:
    similar_tickets = retrieve_similar_tickets(
        ticket_text=ticket_text,
        tickets=tickets,
        embedder=embedder,
        top_k=top_k,
    )
    recommendation = generate_resolution_recommendation(
        ticket_text=ticket_text,
        similar_tickets=similar_tickets,
        model=model,
    )
    return recommendation, similar_tickets
