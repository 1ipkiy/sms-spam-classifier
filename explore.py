"""
Quick EDA for the SMS spam dataset.
Shows class balance and a few example messages (no full dump).

Run:  python explore.py

Author: Miles Robinson (2026)
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset


def main() -> None:
    print("Loading ucirvine/sms_spam ...")
    ds = load_dataset("ucirvine/sms_spam", split="train")
    labels = [int(r["label"]) for r in ds]
    texts = [r["sms"].strip() for r in ds]

    counts = Counter(labels)
    total = len(labels)
    print(f"\nDataset size: {total}")
    print(f"  ham  (0): {counts[0]:5d}  ({100 * counts[0] / total:.1f}%)")
    print(f"  spam (1): {counts[1]:5d}  ({100 * counts[1] / total:.1f}%)")

    lengths = [len(t.split()) for t in texts]
    print(f"\nMessage length (words): min={min(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")

    print("\n--- Example HAM messages ---")
    shown = 0
    for t, y in zip(texts, labels):
        if y == 0:
            print(f"  • {t[:120]}{'...' if len(t) > 120 else ''}")
            shown += 1
            if shown >= 3:
                break

    print("\n--- Example SPAM messages ---")
    shown = 0
    for t, y in zip(texts, labels):
        if y == 1:
            print(f"  • {t[:120]}{'...' if len(t) > 120 else ''}")
            shown += 1
            if shown >= 3:
                break

    print("\nDone. (Full messages stay in the Hugging Face cache — nothing dumped.)")


if __name__ == "__main__":
    main()
