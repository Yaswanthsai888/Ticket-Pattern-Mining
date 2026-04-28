import argparse
import os

import pandas as pd

from rag_pipeline import load_embedder
from rag_pipeline import resolve_ticket


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG-based ticket resolution against a processed dataset."
    )
    parser.add_argument(
        "ticket_text",
        help="New ticket description to analyze.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Dataset directory that contains processed/tickets_clustered.parquet.",
    )
    args = parser.parse_args()

    tickets_path = os.path.join(args.data_dir, "processed", "tickets_clustered.parquet")
    tickets = pd.read_parquet(tickets_path)

    embedder = load_embedder()
    recommendation, similar_tickets = resolve_ticket(
        ticket_text=args.ticket_text,
        tickets=tickets,
        embedder=embedder,
    )

    print("AI Recommended Resolution")
    print("=" * 80)
    print(recommendation)
    print("\nTop Similar Historical Tickets")
    print("=" * 80)
    display_cols = ["Similarity", "Ticket_ID", "Short_Description", "Resolution_Notes"]
    avail_cols = [c for c in display_cols if c in similar_tickets.columns]
    print(similar_tickets[avail_cols].to_string(index=False))


if __name__ == "__main__":
    main()
