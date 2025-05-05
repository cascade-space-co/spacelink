"""
Human-verified equation directive for Sphinx.
"""

from docutils import nodes
from docutils.parsers.rst import Directive


class HumanVerifiedNode(nodes.General, nodes.Element):
    pass


def visit_human_verified_node(self, node):
    self.body.append('<div class="human-verified">')
    self.body.append(
        '<div class="human-verified-title">✓ Human-Verified Equation</div>'
    )


def depart_human_verified_node(self, node):
    self.body.append("</div>")


class HumanVerifiedDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        node = HumanVerifiedNode()
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def setup(app):
    app.add_node(
        HumanVerifiedNode,
        html=(visit_human_verified_node, depart_human_verified_node),
    )
    app.add_directive("human-verified", HumanVerifiedDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
