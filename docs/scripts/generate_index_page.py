from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import mkdocs_gen_files as gen


def read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def extract_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return fallback


def extract_summary(markdown_text: str) -> str:
    # 先頭の見出しをスキップして、最初の非空行の段落を要約として使う
    lines = markdown_text.splitlines()
    # スキップ: 先頭の連続する見出し行(# から始まる)や空行
    content_started = False
    paragraph_lines: list[str] = []
    for line in lines:
        if not content_started:
            if line.strip().startswith("#") or line.strip() == "":
                continue
            content_started = True
        if content_started:
            if line.strip() == "":
                break
            paragraph_lines.append(line.rstrip())
    summary = " ".join(l.strip() for l in paragraph_lines).strip()
    # 長すぎる場合は短縮
    if len(summary) > 160:
        summary = summary[:157].rstrip() + "…"
    return summary


def iter_instance_dirs(instances_root: Path) -> Iterable[Path]:
    for child in sorted(instances_root.iterdir()):
        if child.is_dir() and (child / "description.md").exists():
            yield child


def main() -> None:
    # このスクリプトは docs/scripts/ 以下に置かれるため、プロジェクトルートは parents[2]
    project_root = Path(__file__).resolve().parents[2]
    instances_root = (project_root / "incetances").resolve()
    print(f"[gen] project_root={project_root}")
    print(
        f"[gen] instances_root exists={instances_root.exists()} path={instances_root}"
    )

    cards_items: list[str] = []

    found: list[Path] = list(iter_instance_dirs(instances_root))
    print(f"[gen] found instances: {[p.name for p in found]}")
    for inst_dir in found:
        desc_path = inst_dir / "description.md"
        raw_md = read_text(desc_path)

        slug = inst_dir.name
        title = extract_title(raw_md, fallback=slug)
        summary = extract_summary(raw_md) or ""

        # 出力先: instances/<slug>.md
        dst_page_path = f"instances/{slug}.md"
        print(f"[gen] write page: {dst_page_path}")
        with gen.open(dst_page_path, "w", encoding="utf-8") as f:
            # 元の description.md をそのままコピー（タイトルも含む）
            f.write(raw_md.strip() + "\n")

        # カード用アイテムを組み立て
        # Material の cards レイアウト: <div class="grid cards" markdown> の中で list item を使う
        page_url = f"instances/{slug}/"
        if summary:
            item_line = f'-   <a href="{page_url}"><strong>{title}</strong><br />\n    {summary}</a>'
        else:
            item_line = f'-   <a href="{page_url}"><strong>{title}</strong></a>'
        cards_items.append(item_line)

    # トップページを生成: index.md（既存の index.md があっても生成物で上書き）
    # オプション: 任意に追記できるヘッダー/フッターを取り込む
    header_path = project_root / "docs" / "index_header.md"
    footer_path = project_root / "docs" / "index_footer.md"

    index_lines: list[str] = ["# インスタンス一覧", ""]
    if header_path.exists():
        header_md = read_text(header_path).strip()
        if header_md:
            index_lines.extend([header_md, ""])  # 空行で区切る

    # コンテナでカードレイアウトを適用
    index_lines.append('<div class="grid cards" markdown>')
    index_lines.extend(cards_items)
    index_lines.append("</div>")
    if footer_path.exists():
        footer_md = read_text(footer_path).strip()
        if footer_md:
            index_lines.extend(["", footer_md])
    index_lines.append("")

    # 最低限のページが必ず生成されるようにする
    if not index_lines:
        index_lines = [
            "# インスタンス一覧",
            "",
            "(自動生成) インスタンスが見つかりませんでした。",
        ]
    with gen.open("index.md", "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))


# mkdocs-gen-files はスクリプトをモジュールとして読み込むため、
# import 時に実行されるようにする
main()
