"""
Lightweight progress reporter for the segmentation pipeline.

Sends heartbeat POSTs to an optional API endpoint. If no URL is configured,
all calls are silent no-ops. Failures are logged to stderr but never crash
the pipeline.
"""

import sys
import urllib.request
import urllib.parse


class ProgressReporter:
    """Fire-and-forget progress reporter over HTTP POST.

    Parameters
    ----------
    url : str or None
        Heartbeat endpoint (e.g. ``http://wrapperapi:8080/heartbeat``).
        ``None`` disables all reporting.
    job_id : str or None
        Job identifier sent with each heartbeat.
    auth_header : tuple[str, str] or None
        ``(header_name, header_value)`` added to every request,
        e.g. ``("X-Heartbeat-Token", "abc123")``.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(self, url=None, job_id=None, auth_header=None, timeout=5):
        self.url = url
        self.job_id = job_id or ""
        self.auth_header = auth_header
        self.timeout = timeout

    def report(self, message):
        """Send a progress message.  No-op when *url* is unset."""
        # Always log to console
        print(f"[progress] {message}", flush=True)

        if not self.url:
            return

        try:
            data = urllib.parse.urlencode({
                "jobId": self.job_id,
                "desc": message,
            }).encode("utf-8")

            req = urllib.request.Request(self.url, data=data, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            if self.auth_header:
                req.add_header(self.auth_header[0], self.auth_header[1])

            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:
            print(f"[progress] heartbeat failed: {exc}", file=sys.stderr, flush=True)
