/* WordPiece tokenizer for BERT-family models (bert-tiny / BERT uncased vocab).
 *
 * Loaded as a classic script by the popup and the verification pages, with no
 * bundler. Exports `PhishTokenizer`: cleanText -> basic split -> greedy
 * longest-match WordPiece, matching Hugging Face's BertTokenizer.
 *
 * Deliberately self-contained: no tokenizer library, no CDN, no network. The
 * vocabulary is a plain text file that ships inside the extension.
 */
(function (global) {
  "use strict";

  const CLS_ID = 101;
  const SEP_ID = 102;
  const PAD_ID = 0;
  const UNK_ID = 100;
  const MAX_CHARS_PER_WORD = 100;

  // BERT's punctuation table: the ASCII ranges from BertTokenizer, plus any
  // non-ASCII character whose Unicode category starts with P (’ – … and so on).
  const isPunct = (cp) =>
    (cp >= 33 && cp <= 47) ||
    (cp >= 58 && cp <= 64) ||
    (cp >= 91 && cp <= 96) ||
    (cp >= 123 && cp <= 126) ||
    (cp > 127 && /\p{P}/u.test(String.fromCodePoint(cp)));

  const isSpace = (cp) =>
    cp === 32 || cp === 9 || cp === 10 || cp === 13 || cp === 0x00a0 || cp === 0x2028 || cp === 0x2029;

  const isControl = (cp) => (cp < 32) || cp === 127 || (cp >= 128 && cp <= 159);

  /** Strip combining marks so "café" and "cafe" share a vocabulary entry. */
  function stripAccents(text) {
    try {
      return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    } catch (e) {
      return text;
    }
  }

  function cleanText(text) {
    let out = "";
    for (const ch of String(text || "")) {
      const cp = ch.codePointAt(0);
      // Whitespace (incl. tab/newline/CR) must become a space BEFORE the
      // control-character drop, otherwise words separated only by newlines
      // fuse together - Hugging Face converts them to spaces first.
      if (isSpace(cp)) {
        out += " ";
      } else if (cp === 0 || cp === 0xfffd || isControl(cp)) {
        continue;
      } else {
        out += ch;
      }
    }
    return stripAccents(out.toLowerCase()).replace(/\s+/g, " ").trim();
  }

  // CJK codepoint ranges from Hugging Face BertTokenizer._is_chinese_char:
  // each such character gets its own token instead of fusing into [UNK].
  const isCjk = (cp) =>
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a600 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f);

  /** Split on whitespace and punctuation, exactly like BertTokenizer. */
  function basicTokenize(text) {
    const tokens = [];
    let current = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (isCjk(cp)) {
        if (current) tokens.push(current);
        tokens.push(ch);
        current = "";
      } else if (isSpace(cp)) {
        if (current) tokens.push(current);
        current = "";
      } else if (isPunct(cp)) {
        if (current) tokens.push(current);
        tokens.push(ch);
        current = "";
      } else {
        current += ch;
      }
    }
    if (current) tokens.push(current);
    return tokens;
  }

  class PhishTokenizer {
    constructor(vocabText) {
      this.vocab = new Map();
      const lines = String(vocabText).split(/\r?\n/);
      for (let i = 0; i < lines.length; i++) {
        const tok = lines[i];
        if (tok !== "" || i === 0) this.vocab.set(tok, i);
      }
      this.unkId = this.vocab.has("[UNK]") ? this.vocab.get("[UNK]") : UNK_ID;
    }

    wordpiece(word) {
      if (word.length > MAX_CHARS_PER_WORD) return [this.unkId];
      const ids = [];
      let start = 0;
      while (start < word.length) {
        let end = word.length;
        let found = null;
        while (start < end) {
          let piece = word.slice(start, end);
          if (start > 0) piece = "##" + piece;
          const id = this.vocab.get(piece);
          if (id !== undefined) {
            found = id;
            break;
          }
          end -= 1;
        }
        if (found === null) return [this.unkId];
        ids.push(found);
        start = end;
      }
      return ids;
    }

    /** text -> { inputIds, attentionMask, tokenTypeIds } truncated to maxLen. */
    encode(text, maxLen = 192) {
      const cleaned = cleanText(text);
      const ids = [CLS_ID];
      for (const word of basicTokenize(cleaned)) {
        for (const id of this.wordpiece(word)) ids.push(id);
        if (ids.length >= maxLen - 1) break;
      }
      ids.length = Math.min(ids.length, maxLen - 1);
      ids.push(SEP_ID);

      const len = ids.length;
      const inputIds = new BigInt64Array(maxLen);
      const attentionMask = new BigInt64Array(maxLen);
      const tokenTypeIds = new BigInt64Array(maxLen);
      for (let i = 0; i < len; i++) {
        inputIds[i] = BigInt(ids[i]);
        attentionMask[i] = 1n;
      }
      for (let i = len; i < maxLen; i++) {
        inputIds[i] = BigInt(PAD_ID);
        attentionMask[i] = 0n;
      }
      return { inputIds, attentionMask, tokenTypeIds, length: len };
    }
  }

  global.PhishTokenizer = PhishTokenizer;
  global.PhishTokenizerConstants = { CLS_ID, SEP_ID, PAD_ID, UNK_ID };
})(typeof self !== "undefined" ? self : this);
