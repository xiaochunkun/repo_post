#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch translate repo descriptions EN→ZH using DeepSeek API.

Processes _desc_batch_XX.json files and generates _zh_batch_XX.json.
Supports resume: existing translations in output files are preserved.
After translation, use --apply to insert Chinese into markdown files.

Usage:
    # Translate all batches
    python tools/translate_deepseek.py --api-key sk-xxx

    # Translate one batch (for testing)
    python tools/translate_deepseek.py --api-key sk-xxx --batch 0

    # Apply translations to markdown files
    python tools/translate_deepseek.py --apply

    # Check progress
    python tools/translate_deepseek.py --status
"""

import json
import re
import sys
import time
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

TOOLS_DIR = Path(__file__).resolve().parent
POSTS_DIR = TOOLS_DIR.parent / "docs" / "_posts"
CHUNK_SIZE = 50  # descriptions per API call
MAX_WORKERS = 4  # concurrent API calls

SYSTEM_PROMPT = (
    "你是专业的技术翻译。将以下 GitHub 仓库的英文描述翻译成简洁自然的中文。\n"
    "规则：\n"
    "- 技术专有名词保留英文（React, Docker, Kubernetes, API, CLI, SDK 等）\n"
    "- 项目名、品牌名保留英文\n"
    "- 翻译要简洁，一句话概括\n"
    "- 仅返回 JSON 对象，key 不变，value 为中文翻译\n"
    "- 不要添加任何解释或 markdown 格式"
)


def create_client(api_key):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ── Translation ──────────────────────────────────────────────


def translate_chunk(client, chunk, model="deepseek-chat", retries=3):
    """Translate a dict {key: english} → {key: chinese} via DeepSeek."""
    prompt = json.dumps(chunk, ensure_ascii=False)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            result = json.loads(text)
            usage = resp.usage
            return result, usage.total_tokens if usage else 0
        except Exception as e:
            wait = min(2 ** attempt, 10)
            print(f"    retry {attempt+1}/{retries}: {e} (wait {wait}s)", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(wait)
    return {}, 0


def translate_batch_file(client, batch_num, model, force=False):
    """Translate one batch file with concurrent API calls."""
    src = TOOLS_DIR / f"_desc_batch_{batch_num:02d}.json"
    dst = TOOLS_DIR / f"_zh_batch_{batch_num:02d}.json"

    if not src.exists():
        return 0, 0, 0

    with open(src, "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    # Load existing progress
    existing = {}
    if dst.exists() and not force:
        with open(dst, "r", encoding="utf-8") as f:
            existing = json.load(f)

    todo = {k: v for k, v in descriptions.items() if k not in existing}
    if not todo:
        print(f"  batch {batch_num:02d}: {len(descriptions)} done (skip)")
        return len(descriptions), 0, 0

    print(f"  batch {batch_num:02d}: {len(todo)} to translate, {len(existing)} done")

    items = list(todo.items())
    translated = dict(existing)
    new_count = 0
    total_tokens = 0

    # Build chunks
    chunks = []
    for i in range(0, len(items), CHUNK_SIZE):
        chunks.append(dict(items[i : i + CHUNK_SIZE]))

    total_chunks = len(chunks)

    # Process chunks concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(translate_chunk, client, chunk, model): ci
            for ci, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            ci = futures[future]
            result, tokens = future.result()
            translated.update(result)
            new_count += len(result)
            total_tokens += tokens

            # Save incrementally
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)

            print(f"    [chunk {ci+1}/{total_chunks}] +{len(result)} → {len(translated)}/{len(descriptions)}")

    return len(translated), new_count, total_tokens


# ── Apply translations to markdown ───────────────────────────


def apply_translations(dry_run=False):
    """Insert Chinese translations into docs/_posts/*.md files."""
    all_zh = {}
    batch_files = sorted(TOOLS_DIR.glob("_zh_batch_*.json"))
    for zh_file in batch_files:
        with open(zh_file, "r", encoding="utf-8") as f:
            all_zh.update(json.load(f))

    print(f"Loaded {len(all_zh)} translations from {len(batch_files)} batch files")

    applied = 0
    skipped = 0
    errors = 0

    for filename, zh_desc in sorted(all_zh.items()):
        if not zh_desc or not zh_desc.strip():
            skipped += 1
            continue

        filepath = POSTS_DIR / filename
        if not filepath.exists():
            errors += 1
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split front matter
        parts = content.split("---", 2)
        if len(parts) < 3:
            errors += 1
            continue

        body = parts[2]

        # Already has Chinese in body? skip
        if re.search(r"[\u4e00-\u9fff]", body):
            skipped += 1
            continue

        # Parse body lines, find description, insert translation after it
        lines = body.split("\n")
        new_lines = []
        inserted = False
        found_title = False

        for line in lines:
            new_lines.append(line)
            stripped = line.strip()

            if not stripped:
                continue
            if stripped.startswith("#") and "[" in stripped:
                found_title = True
                continue
            if stripped.startswith("[View"):
                continue

            # First non-empty, non-title, non-link line after title = description
            if found_title and not inserted:
                new_lines.append("")
                new_lines.append(zh_desc)
                inserted = True

        if not inserted:
            # Fallback: insert after first non-empty content line
            new_lines = []
            for line in lines:
                new_lines.append(line)
                stripped = line.strip()
                if (
                    stripped
                    and not stripped.startswith("#")
                    and not stripped.startswith("[")
                    and not inserted
                ):
                    new_lines.append("")
                    new_lines.append(zh_desc)
                    inserted = True

        if inserted:
            if dry_run:
                print(f"  {filename}: {zh_desc[:60]}")
            else:
                new_body = "\n".join(new_lines)
                new_content = "---" + parts[1] + "---" + new_body
                if not new_content.endswith("\n"):
                    new_content += "\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
            applied += 1
        else:
            errors += 1

    print(f"\nApplied: {applied}, Skipped (already has zh): {skipped}, Errors: {errors}")


# ── Status ───────────────────────────────────────────────────


def show_status():
    """Show translation progress."""
    total_desc = 0
    total_zh = 0
    print("Batch  Source  Translated  Status")
    print("─" * 45)

    for i in range(100):
        src = TOOLS_DIR / f"_desc_batch_{i:02d}.json"
        if not src.exists():
            break
        with open(src, "r", encoding="utf-8") as f:
            desc_count = len(json.load(f))
        total_desc += desc_count

        dst = TOOLS_DIR / f"_zh_batch_{i:02d}.json"
        zh_count = 0
        if dst.exists():
            with open(dst, "r", encoding="utf-8") as f:
                zh_count = len(json.load(f))
        total_zh += zh_count

        status = "DONE" if zh_count >= desc_count else f"{zh_count}/{desc_count}"
        bar = "█" * int(zh_count / desc_count * 20) if desc_count else ""
        print(f"  {i:02d}    {desc_count:>5}  {zh_count:>10}  {status:>10}  {bar}")

    pct = total_zh / total_desc * 100 if total_desc else 0
    print("─" * 45)
    print(f"Total: {total_zh}/{total_desc} ({pct:.1f}%)")


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Translate descriptions with DeepSeek")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--batch", type=int, default=-1, help="Specific batch number")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--apply", action="store_true", help="Apply to markdown files")
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument("--force", action="store_true", help="Re-translate existing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.apply:
        apply_translations(dry_run=args.dry_run)
        return

    if not args.api_key:
        print("ERROR: provide --api-key or set DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = create_client(args.api_key)

    # Determine which batches to process
    if args.batch >= 0:
        batches = [args.batch]
    else:
        batches = []
        for i in range(100):
            if (TOOLS_DIR / f"_desc_batch_{i:02d}.json").exists():
                batches.append(i)

    print(f"Processing {len(batches)} batch(es), model={args.model}\n")

    total_done = 0
    total_new = 0
    total_tokens = 0
    start = time.time()

    for batch_num in batches:
        done, new, tokens = translate_batch_file(
            client, batch_num, args.model, force=args.force
        )
        total_done += done
        total_new += new
        total_tokens += tokens

    elapsed = time.time() - start
    print(f"\n{'═' * 45}")
    print(f"Total: {total_done} items, {total_new} newly translated")
    print(f"Tokens used: {total_tokens:,}")
    print(f"Time: {elapsed/60:.1f} min")

    # Cost estimate (DeepSeek-V3 pricing)
    cost = total_tokens / 1_000_000 * 0.5  # rough average $/M tokens
    print(f"Estimated cost: ~${cost:.2f}")


if __name__ == "__main__":
    main()
