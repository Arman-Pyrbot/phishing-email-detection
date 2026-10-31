/* Content script: answers the popup with the current page's visible text.
 *
 * Two shapes break a naive extractor. Application pages (webmail, dashboards)
 * put navigation chrome before the content in DOM order, so a prefix of
 * body.innerText is a list of menu labels; and some of them render the main
 * content inside a frame rather than the top document. So we keep the prose -
 * the direct text of elements that are not inside a link, menu or navigation
 * region - and we read same-origin frames as well as the top document.
 *
 * Classic script (content scripts cannot be ES modules). No banner, no
 * background messaging - extraction happens only when the popup asks.
 */
(() => {
  "use strict";
  const TEXT_LIMIT = 1500;
  const MIN_BLOCK_CHARS = 40;
  const MIN_PROSE_CHARS = 200;
  // Regions whose text is interface chrome rather than page content.
  const CHROME = "a,nav,header,footer,aside,button,select,option,script,style," +
    "noscript,svg,[role=navigation],[role=menu],[role=menubar],[role=tablist]," +
    "[role=link],[role=button],[role=toolbar],[role=banner]";

  /** An element's own words: its direct text nodes, not its children's. */
  function ownText(el) {
    let text = "";
    for (const node of el.childNodes) {
      if (node.nodeType === 3) text += node.nodeValue;
    }
    return text.replace(/\s+/g, " ").trim();
  }

  /** Prose blocks in document order, with navigation chrome left out. */
  function collect(root) {
    const blocks = [];
    for (const el of root.querySelectorAll("*")) {
      if (el.closest(CHROME)) continue;
      const text = ownText(el);
      if (text.length >= MIN_BLOCK_CHARS) blocks.push(text);
    }
    return blocks.join(" ");
  }

  /** The page's prose, including same-origin frames; cross-origin ones are
   *  inaccessible by design and are skipped. */
  function pageProse() {
    if (!document.body) return "";
    let prose = collect(document.body);
    for (const frame of document.querySelectorAll("iframe, frame")) {
      let doc = null;
      try { doc = frame.contentDocument; } catch (e) { doc = null; }
      if (doc && doc.body) prose += " " + collect(doc.body);
    }
    // If that found almost nothing (canvas, link-only markup), the whole page
    // is a better answer than an empty one.
    return prose.length >= MIN_PROSE_CHARS ? prose : document.body.innerText || "";
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || msg.type !== "EXTRACT_PAGE_TEXT") return false;
    let body = "";
    try {
      body = pageProse();
    } catch (e) {
      // Never leave the popup without an answer: fall back to the whole page.
      body = document.body ? document.body.innerText || "" : "";
    }
    sendResponse({
      url: location.href,
      title: (document.title || "").slice(0, 200),
      text: body.replace(/\s+/g, " ").trim().slice(0, TEXT_LIMIT),
    });
    return false; // response is synchronous
  });
})();
