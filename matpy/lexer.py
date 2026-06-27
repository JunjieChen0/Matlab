"""MatPy lexer — tokenizes Matlab source code."""

from matpy.tokens import Token, TokenType, KEYWORDS


class LexError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"Line {line}, Col {col}: {message}")
        self.line = line
        self.col = col


class Lexer:
    """Context-sensitive lexer for MATLAB source code.

    MATLAB lexing is tricky because:
    - Whitespace matters inside matrix literals [1 +2] vs [1+2]
    - ' can be transpose or string delimiter depending on context
    - Numbers like 1./2 are ambiguous
    - Command form: `hold on` == `hold('on')`
    """

    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []
        self._bracket_depth = 0
        self._prev_token: Token | None = None

    # ── helpers ──────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> str | None:
        p = self.pos + offset
        return self.source[p] if p < len(self.source) else None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _add(
        self, ttype: TokenType, value, line: int | None = None, col: int | None = None
    ):
        tok = Token(ttype, value, line or self.line, col or self.col)
        self.tokens.append(tok)
        self._prev_token = tok
        return tok

    def _is_at_end(self) -> bool:
        return self.pos >= len(self.source)

    # ── main entry ───────────────────────────────────────────────

    def tokenize(self) -> list[Token]:
        while not self._is_at_end():
            self._scan_token()
        if self.tokens and self.tokens[-1].type != TokenType.NEWLINE:
            self._add(TokenType.NEWLINE, "\\n")
        self._add(TokenType.EOF, None)
        return self.tokens

    # ── core dispatch ────────────────────────────────────────────

    def _scan_token(self):
        ch = self._peek()

        if ch == " " or ch == "\t":
            self._advance()
            return

        if ch == "\n" or ch == "\r":
            self._advance()
            if ch == "\r" and self._peek() == "\n":
                self._advance()
            if self._prev_token is None or self._prev_token.type != TokenType.NEWLINE:
                self._add(TokenType.NEWLINE, "\\n")
            return

        if ch == "%":
            self._scan_comment()
            return

        if ch == "." and self._peek(1) == "." and self._peek(2) == ".":
            self._scan_ellipsis()
            return

        if ch and ch.isdigit():
            self._scan_number()
            return

        if ch == "." and self._peek(1) and self._peek(1).isdigit():
            self._scan_number()
            return

        if ch == "'":
            self._scan_string_or_transpose()
            return

        if ch == '"':
            self._scan_dq_string()
            return

        if ch and (ch.isalpha() or ch == "_"):
            self._scan_identifier()
            return

        self._scan_operator()

    # ── scanning sub-routines ────────────────────────────────────

    def _scan_comment(self):
        self._advance()  # consume %
        if self._peek() == "{":
            self._advance()
            self._scan_block_comment()
            return
        while not self._is_at_end() and self._peek() != "\n":
            self._advance()

    def _scan_block_comment(self):
        while not self._is_at_end():
            if self._peek() == "%" and self._peek(1) == "}":
                self._advance()
                self._advance()
                return
            self._advance()

    def _scan_ellipsis(self):
        self._advance()  # .
        self._advance()  # .
        self._advance()  # .
        while not self._is_at_end() and self._peek() != "\n":
            self._advance()

    def _scan_number(self):
        line, col = self.line, self.col
        start = self.pos
        has_dot = False
        has_exp = False

        if self._peek() == ".":
            has_dot = True
            self._advance()

        while not self._is_at_end() and self._peek() and self._peek().isdigit():
            self._advance()

        if not self._is_at_end() and self._peek() == "." and self._peek(1) != ".":
            if self._peek(1) and (self._peek(1).isdigit() or self._peek(1) in "eE"):
                has_dot = True
                self._advance()
                while not self._is_at_end() and self._peek() and self._peek().isdigit():
                    self._advance()

        if not self._is_at_end() and self._peek() and self._peek() in "eE":
            has_exp = True
            self._advance()
            if not self._is_at_end() and self._peek() and self._peek() in "+-":
                self._advance()
            while not self._is_at_end() and self._peek() and self._peek().isdigit():
                self._advance()

        if not self._is_at_end() and self._peek() and self._peek() in "ij":
            self._advance()

        text = self.source[start : self.pos]
        try:
            if "i" in text or "j" in text:
                num_text = text.replace("i", "").replace("j", "")
                if num_text in ("", "+", "-"):
                    num_text += "1"
                value = complex(0, float(num_text))
            elif has_dot or has_exp:
                value = float(text)
            else:
                value = int(text)
        except ValueError:
            raise LexError(f"Invalid number: {text}", line, col)

        self._add(TokenType.NUMBER, value, line, col)

    def _scan_string_or_transpose(self):
        line, col = self.line, self.col

        if self._prev_token and self._prev_token.type in (
            TokenType.NUMBER,
            TokenType.STRING,
            TokenType.IDENTIFIER,
            TokenType.RPAREN,
            TokenType.RBRACKET,
            TokenType.RBRACE,
            TokenType.END,
        ):
            self._advance()
            if self._peek() == ".":
                self._advance()
                self._add(TokenType.DOT_TRANSPOSE, ".'", line, col)
            else:
                self._add(TokenType.TRANSPOSE, "'", line, col)
            return

        self._advance()  # consume opening '
        chars = []
        while not self._is_at_end():
            ch = self._peek()
            if ch == "'":
                self._advance()
                if self._peek() == "'":
                    chars.append("'")
                    self._advance()
                else:
                    break
            elif ch == "\n":
                raise LexError("Unterminated string literal", line, col)
            else:
                chars.append(ch)
                self._advance()
        self._add(TokenType.STRING, "".join(chars), line, col)

    def _scan_dq_string(self):
        line, col = self.line, self.col
        self._advance()  # consume opening "
        chars = []
        while not self._is_at_end():
            ch = self._peek()
            if ch == '"':
                self._advance()
                if self._peek() == '"':
                    chars.append('"')
                    self._advance()
                else:
                    break
            elif ch == "\\":
                self._advance()
                esc = self._peek()
                if esc is None:
                    raise LexError(
                        "Unterminated escape sequence at end of string", line, col
                    )
                elif esc == "n":
                    chars.append("\n")
                    self._advance()
                elif esc == "t":
                    chars.append("\t")
                    self._advance()
                elif esc == "\\":
                    chars.append("\\")
                    self._advance()
                elif esc == '"':
                    chars.append('"')
                    self._advance()
                else:
                    chars.append("\\")
            elif ch == "\n":
                raise LexError("Unterminated string literal", line, col)
            else:
                chars.append(ch)
                self._advance()
        self._add(TokenType.STRING, "".join(chars), line, col)

    def _scan_identifier(self):
        line, col = self.line, self.col
        start = self.pos
        while (
            not self._is_at_end()
            and self._peek()
            and (self._peek().isalnum() or self._peek() == "_")
        ):
            self._advance()
        text = self.source[start : self.pos]
        ttype = KEYWORDS.get(text, TokenType.IDENTIFIER)
        self._add(ttype, text, line, col)

    def _scan_operator(self):
        line, col = self.line, self.col
        ch = self._advance()

        match ch:
            case "+":
                self._add(TokenType.PLUS, "+", line, col)
            case "-":
                self._add(TokenType.MINUS, "-", line, col)
            case "(":
                self._add(TokenType.LPAREN, "(", line, col)
            case ")":
                self._add(TokenType.RPAREN, ")", line, col)
            case "[":
                self._bracket_depth += 1
                self._add(TokenType.LBRACKET, "[", line, col)
            case "]":
                self._bracket_depth = max(0, self._bracket_depth - 1)
                self._add(TokenType.RBRACKET, "]", line, col)
            case "{":
                self._bracket_depth += 1
                self._add(TokenType.LBRACE, "{", line, col)
            case "}":
                self._bracket_depth = max(0, self._bracket_depth - 1)
                self._add(TokenType.RBRACE, "}", line, col)
            case ";":
                self._add(TokenType.SEMICOLON, ";", line, col)
            case ",":
                self._add(TokenType.COMMA, ",", line, col)
            case "@":
                self._add(TokenType.AT, "@", line, col)
            case ":":
                self._add(TokenType.COLON, ":", line, col)
            case ".":
                nxt = self._peek()
                if nxt == "*":
                    self._advance()
                    self._add(TokenType.DOT_STAR, ".*", line, col)
                elif nxt == "/":
                    self._advance()
                    self._add(TokenType.DOT_SLASH, "./", line, col)
                elif nxt == "\\":
                    self._advance()
                    self._add(TokenType.DOT_BACKSLASH, ".\\", line, col)
                elif nxt == "^":
                    self._advance()
                    self._add(TokenType.DOT_CARET, ".^", line, col)
                elif nxt == "'":
                    self._advance()
                    self._add(TokenType.DOT_TRANSPOSE, ".'", line, col)
                else:
                    self._add(TokenType.DOT, ".", line, col)
            case "*":
                self._add(TokenType.STAR, "*", line, col)
            case "/":
                self._add(TokenType.SLASH, "/", line, col)
            case "\\":
                self._add(TokenType.BACKSLASH, "\\", line, col)
            case "^":
                self._add(TokenType.CARET, "^", line, col)
            case "=":
                if self._peek() == "=":
                    self._advance()
                    self._add(TokenType.EQ, "==", line, col)
                else:
                    self._add(TokenType.ASSIGN, "=", line, col)
            case "~":
                if self._peek() == "=":
                    self._advance()
                    self._add(TokenType.NEQ, "~=", line, col)
                else:
                    self._add(TokenType.TILDE, "~", line, col)
            case "<":
                if self._peek() == "=":
                    self._advance()
                    self._add(TokenType.LE, "<=", line, col)
                else:
                    self._add(TokenType.LT, "<", line, col)
            case ">":
                if self._peek() == "=":
                    self._advance()
                    self._add(TokenType.GE, ">=", line, col)
                else:
                    self._add(TokenType.GT, ">", line, col)
            case "&":
                if self._peek() == "&":
                    self._advance()
                    self._add(TokenType.AMP_AMP, "&&", line, col)
                else:
                    self._add(TokenType.AMP, "&", line, col)
            case "|":
                if self._peek() == "|":
                    self._advance()
                    self._add(TokenType.PIPE_PIPE, "||", line, col)
                else:
                    self._add(TokenType.PIPE, "|", line, col)
            case _:
                raise LexError(f"Unexpected character: {ch!r}", line, col)
