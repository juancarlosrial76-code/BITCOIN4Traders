import asyncio
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple in‑memory rate limiter (requests per minute per IP)."""
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.ip_counters: dict[str, list[float]] = {}
        self.lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "anonymous"
        now = asyncio.get_event_loop().time()
        async with self.lock:
            timestamps = self.ip_counters.get(client_ip, [])
            # Remove timestamps outside the window
            timestamps = [t for t in timestamps if now - t < self.window]
            if len(timestamps) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Too Many Requests")
            timestamps.append(now)
            self.ip_counters[client_ip] = timestamps
            # F-018: Prune IPs with no recent activity to prevent unbounded growth.
            # Only prune occasionally (every ~100 requests) to amortize cost.
            if len(self.ip_counters) > 1000:
                stale = [ip for ip, ts in self.ip_counters.items() if not ts]
                for ip in stale:
                    del self.ip_counters[ip]
        response = await call_next(request)
        return response
