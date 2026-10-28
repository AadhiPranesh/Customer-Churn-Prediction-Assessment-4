"""
Clean up hardcoded machine-specific 'file:///c:/7th%20Sem/AADHI/' URLs in markdown reports,
replacing them with portable relative paths so documents render cleanly anywhere.
"""

from pathlib import Path
import re

def clean_markdown_links():
    root = Path(__file__).resolve().parents[1]
    md_files = list(root.glob('**/*.md'))
    
    target_pattern = re.compile(r'file:///c:/7th%20Sem/AADHI/', re.IGNORECASE)
    
    count_changed = 0
    for md_path in md_files:
        content = md_path.read_text(encoding='utf-8', errors='ignore')
        if target_pattern.search(content):
            # Compute relative prefix from this md_path's parent directory to root
            try:
                rel_to_root = Path(os.path.relpath(root, md_path.parent)).as_posix()
                if rel_to_root == '.':
                    replacement = ''
                else:
                    replacement = rel_to_root + '/'
            except Exception:
                replacement = ''
                
            # Replace file:///c:/7th%20Sem/AADHI/ with proper relative path
            new_content = target_pattern.sub(replacement, content)
            md_path.write_text(new_content, encoding='utf-8')
            print(f"Updated links in {md_path.relative_to(root)}")
            count_changed += 1

    print(f"Cleaned links in {count_changed} markdown files.")

if __name__ == '__main__':
    import os
    clean_markdown_links()
