"""
Normalized adapter for Angel One SmartAPI.

Market data architecture:
    SmartWebSocketV2 -> thread-safe LTP cache -> application

REST ltpData() is retained only as a throttled fallback.  The trading
application should normally use get_cached_ltp() so it does not repeatedly
hit the REST LTP endpoint and trigger Angel One rate limits.
"""

import logging
import threading
import time
import random
from typing import Any, Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class AngelAdapter:
    # SmartWebSocketV2 documents these live-feed exchange types. Currency/
    # additional master segments are retained for resolution/historical access,
    # but are not falsely mapped to an undocumented websocket exchange type.
    EXCHANGE_TYPE_MAP = {
        "NSE": 1, "NFO": 2, "BSE": 3, "BFO": 4, "MCX": 5,
    }

    EXCHANGE_ALIASES = {
        "NSE_CM": "NSE", "NSE-INDEX": "NSE", "NSE_FO": "NFO", "NSE-FO": "NFO",
        "BSE_CM": "BSE", "BSE-INDEX": "BSE", "BSE_FO": "BFO", "BSE-FO": "BFO",
        "MCX_FO": "MCX", "MCX-FO": "MCX", "CURRENCY": "CDS", "CURRENCY_FO": "CDS",
    }

    @classmethod
    def canonical_exchange(cls, exchange: str) -> str:
        value = str(exchange or "").strip().upper()
        return cls.EXCHANGE_ALIASES.get(value, value)

    def __init__(self, smart_api):
        self.api = smart_api

        self._feed_token: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._api_key: Optional[str] = None
        self._client_code: Optional[str] = None

        self._ltp_cache: Dict[str, Dict[str, Any]] = {}
        self._ltp_lock = threading.RLock()

        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_lock = threading.RLock()
        self._ws_started = False
        self._ws_stop = threading.Event()
        self._subscriptions: Dict[int, set] = {}

        self._rest_ltp_lock = threading.Lock()
        self._last_rest_ltp = 0.0
        self._rest_ltp_min_interval = 1.0

        # One shared orderBook response for a short interval prevents
        # several position threads from independently hammering orderBook().
        self._order_book_lock = threading.Lock()
        self._order_book_cache = None
        self._order_book_cache_time = 0.0
        self._order_book_cache_ttl = 1.0

    # ------------------------------------------------------------------
    # Authentication / feed token
    # ------------------------------------------------------------------
    def getfeed_token(self) -> Optional[str]:
        try:
            token = self.api.getfeedToken()
            if token:
                self._feed_token = str(token)
            return self._feed_token
        except Exception:
            logger.exception("getfeedToken error")
            return None

    def configure_websocket(
        self,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        feed_token: Optional[str] = None,
    ) -> bool:
        if auth_token:
            self._auth_token = str(auth_token)
        if api_key:
            self._api_key = str(api_key)
        if client_code:
            self._client_code = str(client_code)
        if feed_token:
            self._feed_token = str(feed_token)

        missing = []
        if not self._auth_token:
            missing.append("auth_token")
        if not self._api_key:
            missing.append("api_key")
        if not self._client_code:
            missing.append("client_code")
        if not self._feed_token:
            missing.append("feed_token")

        if missing:
            logger.error("WebSocket configuration incomplete: %s", ", ".join(missing))
            return False
        return True

    # ------------------------------------------------------------------
    # SmartWebSocketV2
    # ------------------------------------------------------------------
    def start_market_data(
        self,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        feed_token: Optional[str] = None,
    ) -> bool:
        with self._ws_lock:
            if self._ws_started and self._ws is not None:
                return True

            if not self.configure_websocket(
                auth_token=auth_token,
                api_key=api_key,
                client_code=client_code,
                feed_token=feed_token,
            ):
                return False

            try:
                from SmartApi.smartWebSocketV2 import SmartWebSocketV2
            except Exception:
                logger.exception(
                    "SmartWebSocketV2 import failed. Install/update smartapi-python."
                )
                return False

            try:
                self._ws_stop.clear()
                self._ws = SmartWebSocketV2(
                    self._auth_token,
                    self._api_key,
                    self._client_code,
                    self._feed_token,
                    max_retry_attempt=5,
                    retry_strategy=1,
                    retry_delay=5,
                    retry_multiplier=2,
                    retry_duration=60,
                )

                self._ws.on_open = self._on_ws_open
                self._ws.on_data = self._on_ws_data
                self._ws.on_error = self._on_ws_error
                self._ws.on_close = self._on_ws_close
                if hasattr(self._ws, "on_control_message"):
                    self._ws.on_control_message = self._on_ws_control

                self._ws_started = True
                self._ws_thread = threading.Thread(
                    target=self._ws_worker,
                    name="AngelMarketDataWS",
                    daemon=True,
                )
                self._ws_thread.start()
                logger.info("Angel One SmartWebSocketV2 started")
                return True
            except Exception:
                logger.exception("Failed to create SmartWebSocketV2")
                self._ws = None
                self._ws_thread = None
                self._ws_started = False
                return False

    def _ws_worker(self):
        try:
            self._ws.connect()
        except Exception:
            logger.exception("SmartWebSocketV2 connection failed")
        finally:
            self._ws_started = False

    def _on_ws_open(self, wsapp):
        logger.info("Angel One market-data WebSocket connected")
        try:
            self._resubscribe_all()
        except Exception:
            logger.exception("WebSocket initial subscription failed")

    def _on_ws_data(self, wsapp, message):
        try:
            if not isinstance(message, dict):
                return

            token = message.get("token")
            raw_ltp = message.get("last_traded_price")
            if token is None or raw_ltp is None:
                return

            token = str(token)
            # SmartWebSocketV2 sends prices in paise.
            ltp = float(raw_ltp) / 100.0

            with self._ltp_lock:
                self._ltp_cache[token] = {
                    "ltp": ltp,
                    "token": token,
                    "exchange_type": message.get("exchange_type"),
                    "exchange": self._exchange_name(message.get("exchange_type")),
                    "timestamp": message.get("exchange_timestamp"),
                    "received_at": time.time(),
                    "source": "WEBSOCKET",
                    "raw": message,
                }
        except Exception:
            logger.exception("WebSocket tick processing failed")

    def _on_ws_error(self, wsapp, error):
        logger.error("Angel One WebSocket error: %s", error)

    def _on_ws_close(self, wsapp):
        logger.warning("Angel One WebSocket closed")

    def _on_ws_control(self, wsapp, message):
        logger.debug("WebSocket control message: %s", message)

    def _resubscribe_all(self):
        if not self._ws:
            return

        token_list = []
        for exchange_type, tokens in self._subscriptions.items():
            if tokens:
                token_list.append({
                    "exchangeType": exchange_type,
                    "tokens": list(tokens),
                })

        if token_list:
            self._ws.subscribe(
                "ltpcache01",
                self._ws.LTP_MODE,
                token_list,
            )
            logger.info("Resubscribed %d exchange groups", len(token_list))

    def subscribe_tokens(self, queries: List[Tuple[str, str]]) -> bool:
        if not queries:
            return True
        if not self._ws:
            logger.error("Cannot subscribe: market-data WebSocket is not running")
            return False

        grouped: Dict[int, List[str]] = {}
        for exchange, token in queries:
            exchange_type = self.EXCHANGE_TYPE_MAP.get(self.canonical_exchange(exchange))
            if exchange_type is None:
                logger.warning("Unsupported WebSocket exchange: %s", exchange)
                continue
            token = str(token).strip()
            if token:
                grouped.setdefault(exchange_type, [])
                if token not in grouped[exchange_type]:
                    grouped[exchange_type].append(token)

        if not grouped:
            return False

        token_list = []
        for exchange_type, tokens in grouped.items():
            current = self._subscriptions.setdefault(exchange_type, set())
            new_tokens = [t for t in tokens if t not in current]
            if new_tokens:
                current.update(new_tokens)
                token_list.append({
                    "exchangeType": exchange_type,
                    "tokens": new_tokens,
                })

        if not token_list:
            return True

        try:
            self._ws.subscribe("ltpcache01", self._ws.LTP_MODE, token_list)
            logger.info("Subscribed %d new token groups", len(token_list))
            return True
        except Exception:
            logger.exception("WebSocket subscribe failed")
            return False

    def unsubscribe_tokens(self, queries: List[Tuple[str, str]]) -> bool:
        if not queries or not self._ws:
            return False

        grouped: Dict[int, List[str]] = {}
        for exchange, token in queries:
            exchange_type = self.EXCHANGE_TYPE_MAP.get(self.canonical_exchange(exchange))
            if exchange_type is None:
                continue
            grouped.setdefault(exchange_type, []).append(str(token))

        token_list = [
            {"exchangeType": et, "tokens": tokens}
            for et, tokens in grouped.items()
        ]

        try:
            self._ws.unsubscribe("ltpcache01", self._ws.LTP_MODE, token_list)
            for et, tokens in grouped.items():
                existing = self._subscriptions.get(et, set())
                for token in tokens:
                    existing.discard(token)
            return True
        except Exception:
            logger.exception("WebSocket unsubscribe failed")
            return False

    # ------------------------------------------------------------------
    # LTP cache / fallback
    # ------------------------------------------------------------------
    def get_cached_ltp(self, token: str, max_age: float = 10.0) -> Optional[float]:
        with self._ltp_lock:
            item = self._ltp_cache.get(str(token))
            if not item:
                return None
            if max_age is not None and time.time() - item.get("received_at", 0) > max_age:
                return None
            try:
                return float(item["ltp"])
            except Exception:
                return None

    def get_cached_tick(self, token: str) -> Optional[Dict[str, Any]]:
        with self._ltp_lock:
            item = self._ltp_cache.get(str(token))
            return dict(item) if item else None

    def ltp(
        self,
        exchange: str,
        tradingsymbol: str,
        token: str,
        max_cache_age: float = 10.0,
        allow_rest_fallback: bool = True,
    ) -> Optional[float]:
        cached = self.get_cached_ltp(token, max_age=max_cache_age)
        if cached is not None:
            return cached

        if not allow_rest_fallback:
            return None

        return self._rest_ltp(exchange, tradingsymbol, token)

    def _rest_ltp(self, exchange: str, tradingsymbol: str, token: str) -> Optional[float]:
        for attempt in range(3):
            try:
                with self._rest_ltp_lock:
                    elapsed = time.monotonic() - self._last_rest_ltp
                    if elapsed < self._rest_ltp_min_interval:
                        time.sleep(self._rest_ltp_min_interval - elapsed)
                    self._last_rest_ltp = time.monotonic()
                    resp = self.api.ltpData(exchange, tradingsymbol, token)

                if not isinstance(resp, dict):
                    return None
                data = resp.get("data")
                if not isinstance(data, dict) or data.get("ltp") is None:
                    return None

                value = float(data["ltp"])
                with self._ltp_lock:
                    self._ltp_cache[str(token)] = {
                        "ltp": value,
                        "token": str(token),
                        "exchange": exchange,
                        "tradingsymbol": tradingsymbol,
                        "received_at": time.time(),
                        "source": "REST",
                        "raw": resp,
                    }
                return value

            except Exception as e:
                msg = str(e).lower()
                rate_limited = (
                    "exceeding access rate" in msg
                    or "access denied" in msg
                    or "rate limit" in msg
                    or "too many requests" in msg
                )
                if rate_limited:
                    delay = min(10.0, 2.0 * (2 ** attempt)) + random.uniform(0.2, 1.0)
                    logger.warning(
                        "Angel One LTP REST rate limit; retry %d/3 in %.2fs",
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                logger.exception("REST LTP fallback failed")
                return None

        logger.error("REST LTP fallback exhausted for %s", tradingsymbol)
        return None

    def ltp_batch(self, queries: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str], float]:
        out = {}
        for exchange, symbol, token in queries:
            value = self.ltp(exchange, symbol, token)
            if value is not None:
                out[(exchange, symbol)] = value
        return out

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = self.api.placeOrder(params)
            if isinstance(resp, dict):
                status = resp.get("status")
                data = resp.get("data")
                if data is not None:
                    if isinstance(data, str):
                        orderid = data
                    elif isinstance(data, dict):
                        orderid = (
                            data.get("orderid")
                            or data.get("order_id")
                            or data.get("orderId")
                            or str(data)
                        )
                    else:
                        orderid = str(data)
                    self.invalidate_order_book_cache()
                    return {"ok": True, "broker_order_id": str(orderid), "raw": resp}
                if status in (True, "success"):
                    return {"ok": True, "broker_order_id": None, "raw": resp}
            self.invalidate_order_book_cache()
            return {"ok": False, "broker_order_id": None, "raw": resp}
        except Exception as e:
            logger.exception("place_order error")
            return {"ok": False, "broker_order_id": None, "raw": str(e)}

    def invalidate_order_book_cache(self):
        with self._order_book_lock:
            self._order_book_cache = None
            self._order_book_cache_time = 0.0

    def cancel_order(self, broker_order_id: str) -> Dict[str, Any]:
        try:
            resp = self.api.cancelOrder({"orderid": str(broker_order_id)})
            ok = resp.get("status") in (True, "success") if isinstance(resp, dict) else True
            self.invalidate_order_book_cache()
            return {"ok": ok, "raw": resp}
        except Exception as e:
            logger.exception("cancel_order error")
            return {"ok": False, "raw": str(e)}

    def positions(self) -> Optional[Dict[str, Any]]:
        """Return Angel One's authoritative current position book."""
        try:
            return self.api.position()
        except Exception:
            logger.exception("position error")
            return None

    def trade_book(self) -> Optional[Dict[str, Any]]:
        """Return today's broker trade book for realized-fill reconciliation."""
        try:
            return self.api.tradeBook()
        except Exception:
            logger.exception("trade_book error")
            return None

    def order_book(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        try:
            with self._order_book_lock:
                now = time.monotonic()
                if (
                    not force_refresh
                    and self._order_book_cache is not None
                    and now - self._order_book_cache_time < self._order_book_cache_ttl
                ):
                    return self._order_book_cache

                response = self.api.orderBook()
                self._order_book_cache = response
                self._order_book_cache_time = now
                return response
        except Exception:
            logger.exception("order_book error")
            return None

    def get_candle_data(
        self,
        exchange: str,
        token: str,
        interval: str,
        fromdate: str,
        todate: str,
    ) -> Optional[Dict[str, Any]]:
        exchange = self.canonical_exchange(exchange)
        payload = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": str(interval),
            "fromdate": str(fromdate),
            "todate": str(todate),
        }
        try:
            resp = self.api.getCandleData(payload)
            if not isinstance(resp, dict) or not resp.get("status") or not resp.get("data"):
                logger.warning("Historical API returned no usable data for %s/%s/%s", exchange, token, interval)
                return None
            return resp
        except Exception as exc:
            logger.warning("get_candle_data failed for %s/%s/%s: %s", exchange, token, interval, exc)
            return None

    # ------------------------------------------------------------------
    # Shutdown / helpers
    # ------------------------------------------------------------------
    def stop_market_data(self):
        self._ws_stop.set()
        with self._ws_lock:
            try:
                if self._ws:
                    self._ws.close_connection()
            except Exception:
                logger.exception("WebSocket close error")
            finally:
                self._ws = None
                self._ws_started = False

    def close(self):
        self.stop_market_data()

    def is_market_data_running(self) -> bool:
        return bool(self._ws_started and self._ws is not None)

    @classmethod
    def _exchange_name(cls, exchange_type: Any) -> Optional[str]:
        reverse = {
            1: "NSE",
            2: "NFO",
            3: "BSE",
            4: "BFO",
            5: "MCX",
            7: "NCX",
            13: "CDE",
        }
        try:
            return reverse.get(int(exchange_type))
        except Exception:
            return None
