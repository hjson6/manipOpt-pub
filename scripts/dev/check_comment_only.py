"""Check that the working tree differs from a git ref only in comments.

Python: the ASTs must match with docstrings removed. XML/URDF/SRDF: the
text must match with <!-- --> comments and whitespace removed (and no
comment may contain "--"). Shell/YAML: the non-comment lines must match.
usage: python scripts/dev/check_comment_only.py [ref, default HEAD]
"""
import ast
import re
import subprocess
import sys

ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
changed = subprocess.run(["git", "diff", "--name-only", ref], capture_output=True, text=True,
                         check=True).stdout.split()


def strip_docstrings(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def xml_body(text):
    return re.sub(r"\s+", "", re.sub(r"<!--.*?-->", "", text, flags=re.S))


def hash_body(text):
    return [line.split(" #")[0].rstrip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


bad = 0
for path in changed:
    try:
        new = open(path).read()
    except FileNotFoundError:
        print(f"DELETED  {path}")
        bad += 1
        continue
    old = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True).stdout
    if path.endswith(".py"):
        same = strip_docstrings(ast.parse(old)) == strip_docstrings(ast.parse(new))
    elif path.endswith((".xml", ".urdf", ".srdf", ".xacro")):
        same = xml_body(old) == xml_body(new)
        if any("--" in c for c in re.findall(r"<!--(.*?)-->", new, flags=re.S)):
            print(f"XML '--' {path}")
            bad += 1
    elif path.endswith((".sh", ".yaml", ".yml")):
        same = hash_body(old) == hash_body(new)
    else:
        continue  # docs etc.
    if not same:
        print(f"CODE CHANGED  {path}")
        bad += 1
print("OK: comment-only changes" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
