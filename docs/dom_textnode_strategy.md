# DOM TextNode Strategy (Browser Extension)

This converter is string-based. In real HTML, text is often split across
multiple TextNodes, so direct cross-node regex is fragile.

## Problem

Example source:

```html
這<span class="highlight">不</span>是
```

DOM traversal sees 3 TextNodes:

1. `這`
2. `不`
3. `是`

A rule like `不是 -> 毋是` will miss if you process each node independently.

## Recommended Integration Contract

1. Build processing blocks from contiguous inline TextNodes.
2. Join block text into one buffer.
3. Run converter on that buffer.
4. Map converted text back to original nodes by character offsets.
5. Skip nodes in editable contexts (`input`, `textarea`, `contenteditable`).

## Minimal JS Skeleton

```js
function collectInlineTextBlocks(root) {
  const blocks = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (parent.closest('input, textarea, [contenteditable=\"true\"]')) {
        return NodeFilter.FILTER_REJECT;
      }
      if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let current = [];
  let prev = null;
  while (walker.nextNode()) {
    const n = walker.currentNode;
    const adjacent = prev && prev.parentElement === n.parentElement && prev.nextSibling === n;
    if (!adjacent && current.length) {
      blocks.push(current);
      current = [];
    }
    current.push(n);
    prev = n;
  }
  if (current.length) blocks.push(current);
  return blocks;
}

function rewriteBlock(nodes, convertFn) {
  const offsets = [];
  let src = '';
  for (const node of nodes) {
    offsets.push([src.length, src.length + node.nodeValue.length]);
    src += node.nodeValue;
  }

  const out = convertFn(src);
  if (out === src) return;

  // Simple rebalance strategy: write all output to first node, clear rest.
  // For production UX, prefer proportional split by original lengths.
  nodes[0].nodeValue = out;
  for (let i = 1; i < nodes.length; i += 1) nodes[i].nodeValue = '';
}
```

## Why This Works Better

- Phrase-level rules can match across original tag boundaries.
- You avoid brittle assumptions that text is in one node.
- You keep a deterministic mapping layer between DOM and converter output.
