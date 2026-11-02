"""HTTP-level checks for the demo server's static routes (no torch needed)."""
import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from gnn_aoi.server import Handler


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)  # http.client sends the path verbatim (no "../" folding)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


@pytest.mark.parametrize("path", [
    "/results/figures/../../gnn_aoi/server.py",
    "/results/figures/../../../../../../etc/passwd",
    "/results/figures/%2e%2e/%2e%2e/gnn_aoi/server.py",
])
def test_figures_route_blocks_path_traversal(server, path):
    status, body = get(server, path)
    assert status == 404
    assert b"import" not in body and b"root:" not in body


def test_missing_file_returns_404_not_a_dropped_connection(server):
    status, _ = get(server, "/definitely-not-here.html")
    assert status == 404


def test_health(server):
    assert get(server, "/health") == (200, b"ok")
