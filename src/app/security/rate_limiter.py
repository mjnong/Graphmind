"""Rate limiting utilities for API endpoints."""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, HTTPException
import redis.asyncio as redis

from src.app.configs.config import get_config


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(status_code=429, detail=detail)
        self.retry_after = retry_after


class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window algorithm."""
    
    def __init__(self):
        self.config = get_config()
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """
        Check if request is allowed for the given identifier.
        
        Args:
            identifier: Unique identifier (usually IP address)
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        async with self._lock:
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            # Clean old requests
            self.requests[identifier] = deque(
                timestamp for timestamp in self.requests[identifier]
                if timestamp > window_start
            )
            
            # Check current request count
            current_requests = len(self.requests[identifier])
            
            if current_requests >= self.config.rate_limit_per_minute:
                # Calculate retry after
                oldest_request = self.requests[identifier][0]
                retry_after = int(oldest_request + 60 - now)
                return False, max(retry_after, 1)
            
            # Allow request and record it
            self.requests[identifier].append(now)
            return True, 0
    
    async def cleanup_old_entries(self):
        """Clean up old entries to prevent memory leaks."""
        async with self._lock:
            now = time.time()
            window_start = now - 60
            
            # Remove identifiers with no recent requests
            to_remove = []
            for identifier, timestamps in self.requests.items():
                # Remove old timestamps
                while timestamps and timestamps[0] <= window_start:
                    timestamps.popleft()
                
                # If no recent requests, mark for removal
                if not timestamps:
                    to_remove.append(identifier)
            
            for identifier in to_remove:
                del self.requests[identifier]


class RedisRateLimiter:
    """Redis-based rate limiter with better performance and persistence."""
    
    def __init__(self):
        self.config = get_config()
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.config.redis_url)
        return self._redis
    
    async def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """
        Check if request is allowed using Redis sliding window.
        
        Args:
            identifier: Unique identifier (usually IP address)
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        try:
            r = await self._get_redis()
            
            # Use Redis sliding window pattern
            now = int(time.time())
            window_start = now - 60  # 1 minute window
            key = f"rate_limit:{identifier}"
            
            # Remove old entries and count current requests
            pipe = r.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.expire(key, 120)  # Expire after 2 minutes
            
            results = await pipe.execute()
            current_count = results[1]
            
            if current_count >= self.config.rate_limit_per_minute:
                # Get the oldest entry to calculate retry_after
                oldest_entries = await r.zrange(key, 0, 0, withscores=True)
                if oldest_entries:
                    oldest_time = int(oldest_entries[0][1])
                    retry_after = oldest_time + 60 - now
                    return False, max(retry_after, 1)
                else:
                    return False, 60
            
            # Add current request
            await r.zadd(key, {f"{now}:{identifier}": now})
            
            return True, 0
            
        except Exception as e:
            # Fallback to allowing request if Redis fails
            print(f"Redis rate limiter error: {e}")
            return True, 0
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class RateLimiter:
    """Main rate limiter that can use either in-memory or Redis backend."""
    
    def __init__(self, use_redis: bool = True):
        self.config = get_config()
        self.use_redis = use_redis
        
        if use_redis:
            self.limiter = RedisRateLimiter()
        else:
            self.limiter = InMemoryRateLimiter()
        
        # Cleanup task for in-memory limiter
        if not use_redis:
            asyncio.create_task(self._cleanup_loop())
    
    async def check_rate_limit(self, request: Request) -> None:
        """
        Check rate limit for the given request.
        
        Args:
            request: FastAPI request object
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Check if request is allowed
        allowed, retry_after = await self.limiter.is_allowed(client_ip)
        
        if not allowed:
            raise RateLimitExceeded(
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                retry_after=retry_after
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def _cleanup_loop(self):
        """Background cleanup task for in-memory limiter."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                if isinstance(self.limiter, InMemoryRateLimiter):
                    await self.limiter.cleanup_old_entries()
            except Exception as e:
                print(f"Rate limiter cleanup error: {e}")
    
    async def close(self):
        """Close rate limiter resources."""
        if isinstance(self.limiter, RedisRateLimiter):
            await self.limiter.close()


# Global rate limiter instance
_rate_limiter_instance = None

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter(use_redis=True)
    return _rate_limiter_instance


# Dependency for FastAPI
async def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate limiting."""
    limiter = get_rate_limiter()
    await limiter.check_rate_limit(request)
