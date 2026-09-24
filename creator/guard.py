"""One check in front of every route that writes, queues or spawns.

ComfyUI has no accounts, so its custom nodes have nothing to authenticate
against — and a browser tab on any site can POST to `localhost:8188`. A
*simple* request (a form or `text/plain` body) goes without a CORS preflight,
and `request.json()` parses the body whatever its Content-Type says, so an
attacker page could fire `/continuity/delete` at a file in output/. The
registry's review (policy v0.2, rule 11) asks for two things in place of the
authentication that does not exist: make every legitimate call a non-simple
request, and check the request's Origin against its Host. That is all this
does.

**Non-simple.** A `Content-Type: application/json` body, or an
`X-Continuity-Request` header, is something a browser will only send after a
preflight — which a ComfyUI started without `--enable-cors-header` answers
with nothing, so the POST never leaves the browser. Every JSON route sends the
first; the two multipart uploads (`/refmod/upload`, `/blockout/frames`) cannot
name their body JSON, so they send the second. Anything else is refused (415)
before the handler reads a byte.

**Same origin.** A request that names an Origin must name this server's Host,
or it is refused (403). This is what holds when CORS *is* enabled: the
preflight passes and the browser sends the Origin it is acting for. A request
with no Origin — curl, a script, the desktop app — is passed by this rule,
because the first rule already stops the only caller that could not choose
its headers.

Not authentication: anyone who can reach the port with a client of their own
can still call every route. That is ComfyUI's own posture, and this guard does
not claim more than it closes.
"""

import functools
from urllib.parse import urlsplit

from aiohttp import web

# Either header makes the request non-simple; the multipart uploads use the second.
REQUEST_HEADER = "X-Continuity-Request"
_DEFAULT_PORT = {"http": 80, "https": 443}


def _authority(origin):
    """`scheme://host[:port]` -> `(host, port)` with the scheme's default port
    filled in; `None` when it is not one (an opaque `null` origin, or garbage)."""
    try:
        parts = urlsplit(origin)
        host, port = parts.hostname, parts.port
    except ValueError:   # an unclosed IPv6 bracket, a port that is not a number
        return None
    if parts.scheme not in _DEFAULT_PORT or not host:
        return None
    return host, port or _DEFAULT_PORT[parts.scheme]


def _same_origin(origin, host):
    """Does the Origin name the Host? A Host without a port means the port of
    the Origin's scheme — which is what the browser resolved it to, and is the
    right answer behind a TLS-terminating proxy too."""
    came_from = _authority(origin)
    if came_from is None:
        return False
    serving = _authority(f"{urlsplit(origin).scheme}://{host}")
    return serving is not None and came_from == serving


def _non_simple(request):
    if request.headers.get(REQUEST_HEADER):
        return True
    kind = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    return kind == "application/json"


def same_origin(handler):
    """Refuse a cross-site request before the handler runs. Sits between
    `@routes.post(...)` and the function; the decorated name still calls the
    handler, with the original at `__wrapped__`."""
    @functools.wraps(handler)
    async def guarded(request):
        if not _non_simple(request):
            return web.json_response(
                {"error": f"send application/json, or the {REQUEST_HEADER} header"}, status=415)
        origin = request.headers.get("Origin")
        if origin is not None and not _same_origin(origin, request.headers.get("Host", "")):
            return web.json_response({"error": "that request came from another site"}, status=403)
        return await handler(request)
    return guarded
