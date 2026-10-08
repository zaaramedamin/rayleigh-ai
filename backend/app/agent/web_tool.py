"""The tool that reads a web page. The owner is asked before every page, and sees the address.

The page is untrusted content from the internet, so the tool is built to bring back as little as
possible and to be hard to turn against this computer:

- Plain GET requests to http or https addresses only, on the ordinary ports, with no cookies, no
  login, no script run, and no address with a user name or password in it.
- Only public internet addresses. The name is resolved here and every address must be a public one:
  not this computer, not the home or office network, not the link-local range where cloud
  services keep their credentials. The connection is then made to the address that was checked,
  not to the name again, so a name that changes its answer cannot slip past.
- Redirects are followed by hand, a few at most, to the same site only; a redirect to another site
  is reported so the owner is asked about that address too.
- Time and size limits, and only text types (web pages, plain text, JSON, XML, CSV, Markdown).
- What comes back is the visible text of the page (scripts, styles and hidden parts removed), with
  no links or images, cut to a limit. It is untrusted data to whatever reads it.

The network itself (`resolver`, `requester`) is passed in, so the tests never touch the internet.
"""

import http.client
import ipaddress
import socket
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from app.agent.tools import Param, Tool, ToolError

MAX_URL_CHARS = 500
MAX_BODY_BYTES = 1_000_000
MAX_REDIRECTS = 3
TIMEOUT_SECONDS = 15.0
MAX_PAGE_CHARS = 6000
USER_AGENT = "Reyleight-agent/1 (local assistant; text only)"
TEXT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "text/xml",
    "application/xml",
)
_PORTS = {"http": 80, "https": 443}
_LOCAL_NAME_ENDINGS = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet")


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # keys in lower case
    body: bytes  # at most MAX_BODY_BYTES


Resolver = Callable[[str, int], list[str]]
# (scheme, host, address to connect to, port, path and query) -> the response.
Requester = Callable[[str, str, str, int, str], HttpResponse]


# --- checking an address ------------------------------------------------------------------------


def is_public(address: str) -> bool:
    """True only for an address on the public internet."""
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_url(url: str) -> tuple[str, str, int, str]:
    """(scheme, host, port, path and query) of a URL that may be fetched. Raises ToolError."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ToolError("That is not a valid web address.") from exc
    scheme = parts.scheme.lower()
    if scheme not in _PORTS:
        raise ToolError("Only http:// and https:// addresses are read.")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise ToolError("That web address has no site name.")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ToolError("Addresses with a user name or password in them are never read.")
    if port is not None and port != _PORTS[scheme]:
        raise ToolError(f"Only the ordinary web ports are used, not {port}.")
    if (
        host == "localhost"
        or host.endswith(_LOCAL_NAME_ENDINGS)
        or "." not in host
        and ":" not in host
    ):
        raise ToolError("That is a name on this computer or its network, which is never read.")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return scheme, host, _PORTS[scheme], path


def _real_resolver(host: str, port: int) -> list[str]:
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise ToolError(f"The site {host} could not be found.") from exc
    return list(dict.fromkeys(str(item[4][0]) for item in found))


def public_address(host: str, port: int, resolver: Resolver) -> str:
    """The address to connect to for `host`, if every address it has is public."""
    addresses = resolver(host, port)
    if not addresses:
        raise ToolError(f"The site {host} could not be found.")
    if not all(is_public(a) for a in addresses):
        raise ToolError(f"{host} leads to this computer or a private network, which is never read.")
    return addresses[0]


# --- the real network --------------------------------------------------------------------------


class _Pinned(http.client.HTTPConnection):
    """Connects to a fixed address, whatever the name says."""

    def __init__(self, host: str, address: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedTLS(http.client.HTTPSConnection):
    """Connects to a fixed address and checks the certificate against the site's name."""

    def __init__(self, host: str, address: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)  # type: ignore[attr-defined]


def real_requester(scheme: str, host: str, address: str, port: int, target: str) -> HttpResponse:
    connection: http.client.HTTPConnection = (
        _PinnedTLS(host, address, port, TIMEOUT_SECONDS)
        if scheme == "https"
        else _Pinned(host, address, port, TIMEOUT_SECONDS)
    )
    try:
        connection.request(
            "GET",
            target,
            headers={
                "Host": host,
                "User-Agent": USER_AGENT,
                "Accept": ", ".join(TEXT_TYPES),
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        body = response.read(MAX_BODY_BYTES + 1)
        headers = {key.lower(): value for key, value in response.getheaders()}
        return HttpResponse(response.status, headers, body[:MAX_BODY_BYTES])
    except ssl.SSLError as exc:
        raise ToolError(f"The site {host} has a certificate that cannot be trusted.") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ToolError(f"The site {host} did not answer ({exc.__class__.__name__}).") from exc
    finally:
        connection.close()


# --- turning a page into text -----------------------------------------------------------------

_SKIPPED = {"script", "style", "noscript", "template", "svg", "head", "iframe", "object", "embed",
            "canvas", "select", "option", "button", "form", "nav", "footer", "aside"}  # fmt: skip
_VOID = {
    "br",
    "hr",
    "img",
    "input",
    "meta",
    "link",
    "area",
    "base",
    "col",
    "source",
    "track",
    "wbr",
}
_BREAKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
           "table", "ul", "ol", "blockquote", "pre", "dd", "dt", "main", "header"}  # fmt: skip


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.pieces: list[str] = []
        self._hidden: list[bool] = []  # one entry per open element
        self._in_title = False

    def _is_hidden(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in _SKIPPED:
            return True
        data = {k.lower(): (v or "").lower().replace(" ", "") for k, v in attrs}
        style = data.get("style", "")
        return (
            "hidden" in data
            or data.get("aria-hidden") == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _VOID:
            if tag in _BREAKS and not any(self._hidden):
                self.pieces.append("\n")
            return
        self._hidden.append(self._is_hidden(tag, attrs))
        if tag in _BREAKS and not any(self._hidden):
            self.pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _VOID:
            return
        if self._hidden:
            self._hidden.pop()
        if tag in _BREAKS and not any(self._hidden):
            self.pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not any(self._hidden):
            self.pieces.append(data)


def visible_text(html: str) -> tuple[str, str]:
    """(title, the visible text) of a web page, with scripts, styles and hidden parts removed."""
    parser = _VisibleText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - a page the parser cannot read is read as far as it goes
        pass
    lines = (" ".join(line.split()) for line in "".join(parser.pieces).splitlines())
    text = "\n".join(line for line in lines if line)
    return " ".join(parser.title.split()), text


def _decode(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            charset = value.strip("\"' ")
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


# --- the tool ----------------------------------------------------------------------------------


def fetch_page(
    url: str, resolver: Resolver = _real_resolver, requester: Requester = real_requester
) -> str:
    """The text of the page at `url`, with a first line saying where it came from."""
    current = url
    first_host: str | None = None
    for _ in range(MAX_REDIRECTS + 1):
        scheme, host, port, target = check_url(current)
        if first_host is None:
            first_host = host
        elif host != first_host:
            raise ToolError(
                f"The page redirects to another site: {current}. Ask for that address if you want "
                "it; the owner is asked about each site."
            )
        response = requester(scheme, host, public_address(host, port, resolver), port, target)
        if response.status in (301, 302, 303, 307, 308):
            where = response.headers.get("location", "")
            if not where:
                raise ToolError("The page redirects without saying where.")
            current = urljoin(f"{scheme}://{host}{target}", where)
            continue
        if response.status != 200:
            raise ToolError(f"The site answered with status {response.status}, not the page.")
        content_type = response.headers.get("content-type", "").lower()
        kind = content_type.split(";")[0].strip()
        if kind not in TEXT_TYPES:
            raise ToolError(
                f"That is not a text page ({kind or 'unknown type'}), so it is not read."
            )
        text = _decode(response.body, content_type)
        title = ""
        if kind in ("text/html", "application/xhtml+xml"):
            title, text = visible_text(text)
        head = f"Web page: {scheme}://{host}{target}"
        if title:
            head += f"\nTitle: {title}"
        if not text.strip():
            return f"{head}\n\n(the page has no readable text)"
        return f"{head}\n\n{text}"
    raise ToolError("The page redirects too many times.")


def fetch_web_page_tool(
    resolver: Resolver | None = None, requester: Requester | None = None
) -> Tool:
    def run(args: Mapping[str, Any]) -> str:
        return fetch_page(args["url"], resolver or _real_resolver, requester or real_requester)

    return Tool(
        name="fetch_web_page",
        description=(
            "Read the text of one public web page. Give the full address (https://...). It "
            "returns only the visible text, no links or images, and cannot log in. The user is "
            "asked to approve each page, and it must be a different address to read another site."
        ),
        level="external_read",
        params={"url": Param("string", "The full web address.", max_length=MAX_URL_CHARS)},
        run=run,
        describe=lambda args: f"Read the web page {args['url']} (this uses the internet)",
        max_result_chars=MAX_PAGE_CHARS,
    )
