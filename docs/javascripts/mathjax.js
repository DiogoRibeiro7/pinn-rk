// MathJax configuration for pymdownx.arithmatex in "generic" mode.
//
// Arithmatex wraps maths in \(...\) and \[...\] inside elements carrying the
// "arithmatex" class, so MathJax is pointed at those delimiters and told to ignore the
// rest of the page. Without this file the theory pages render their LaTeX as raw text.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Material for MkDocs swaps page content without a full reload, so re-typeset on
// navigation rather than only on first load.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
