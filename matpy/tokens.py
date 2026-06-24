"""Token type definitions for the MatPy lexer."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()

    # Keywords
    FUNCTION = auto()
    END = auto()
    IF = auto()
    ELSEIF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    SWITCH = auto()
    CASE = auto()
    OTHERWISE = auto()
    TRY = auto()
    CATCH = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    GLOBAL = auto()
    PERSISTENT = auto()
    CLASSDEF = auto()
    PROPERTIES = auto()
    METHODS = auto()
    EVENTS = auto()
    ENUMERATION = auto()
    ARGUMENTS = auto()

    # Arithmetic operators
    PLUS = auto()         # +
    MINUS = auto()        # -
    STAR = auto()         # *
    SLASH = auto()        # /
    BACKSLASH = auto()    # \
    CARET = auto()        # ^
    DOT_STAR = auto()     # .*
    DOT_SLASH = auto()    # ./
    DOT_BACKSLASH = auto()  # .\
    DOT_CARET = auto()    # .^

    # Comparison operators
    EQ = auto()           # ==
    NEQ = auto()          # ~=
    LT = auto()           # <
    GT = auto()           # >
    LE = auto()           # <=
    GE = auto()           # >=

    # Logical operators
    AMP = auto()          # &
    PIPE = auto()         # |
    TILDE = auto()        # ~
    AMP_AMP = auto()      # &&
    PIPE_PIPE = auto()    # ||

    # Assignment
    ASSIGN = auto()       # =

    # Delimiters
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    SEMICOLON = auto()    # ;
    COMMA = auto()        # ,
    COLON = auto()        # :
    DOT = auto()          # .
    AT = auto()           # @

    # Special
    TRANSPOSE = auto()    # '  (conjugate transpose)
    DOT_TRANSPOSE = auto()  # .'
    ELLIPSIS = auto()     # ...
    NEWLINE = auto()

    # End of file
    EOF = auto()


# Map from keyword strings to token types
KEYWORDS: dict[str, TokenType] = {
    "function": TokenType.FUNCTION,
    "end": TokenType.END,
    "if": TokenType.IF,
    "elseif": TokenType.ELSEIF,
    "else": TokenType.ELSE,
    "for": TokenType.FOR,
    "while": TokenType.WHILE,
    "switch": TokenType.SWITCH,
    "case": TokenType.CASE,
    "otherwise": TokenType.OTHERWISE,
    "try": TokenType.TRY,
    "catch": TokenType.CATCH,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "global": TokenType.GLOBAL,
    "persistent": TokenType.PERSISTENT,
    "classdef": TokenType.CLASSDEF,
    "properties": TokenType.PROPERTIES,
    "methods": TokenType.METHODS,
    "events": TokenType.EVENTS,
    "enumeration": TokenType.ENUMERATION,
    "arguments": TokenType.ARGUMENTS,
}


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"
