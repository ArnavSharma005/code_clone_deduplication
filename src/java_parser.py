"""Tree-sitter based Java method extraction.

Used both to build the training/index corpus from raw .java files and,
later, by the Claude Code hook to pull out the function(s) touched by an
edit before embedding them.
"""
from dataclasses import dataclass

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

JAVA_LANGUAGE = Language(tsjava.language())
_parser = Parser(JAVA_LANGUAGE)

_METHOD_TYPES = ("method_declaration", "constructor_declaration")


@dataclass
class JavaMethod:
    name: str
    code: str
    start_line: int
    end_line: int
    class_name: str | None


def _find_enclosing_class(node: Node) -> str | None:
    cur = node.parent
    while cur is not None:
        if cur.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            for child in cur.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")
        cur = cur.parent
    return None


def _method_name(node: Node) -> str:
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return "<anonymous>"


def extract_methods(source: str | bytes) -> list[JavaMethod]:
    """Parse Java source and return every method/constructor body as a JavaMethod."""
    if isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source

    tree = _parser.parse(source_bytes)
    methods: list[JavaMethod] = []

    def walk(node: Node):
        if node.type in _METHOD_TYPES and node.child_by_field_name("body") is not None:
            methods.append(
                JavaMethod(
                    name=_method_name(node),
                    code=node.text.decode("utf-8"),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    class_name=_find_enclosing_class(node),
                )
            )
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return methods


def extract_methods_from_file(path: str) -> list[JavaMethod]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return extract_methods(f.read())
