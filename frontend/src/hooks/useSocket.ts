import { useCallback, useEffect, useRef, useState } from "react";

export interface SocketMessage {
  type: string;
  [key: string]: unknown;
}

interface UseSocketReturn {
  isConnected: boolean;
  lastMessage: SocketMessage | null;
  messageHistory: SocketMessage[];
  sendMessage: (msg: Record<string, unknown>) => void;
  clearHistory: () => void;
}

export function useSocket(
  url: string,
  reconnectDelay: number = 3000
): UseSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<SocketMessage | null>(null);
  const [messageHistory, setMessageHistory] = useState<SocketMessage[]>([]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.info("[WS] Connected →", url);
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.info("[WS] Disconnected. Retrying in", reconnectDelay, "ms");
      reconnectRef.current = setTimeout(connect, reconnectDelay);
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as SocketMessage;
        setLastMessage(msg);
        setMessageHistory((prev) => [...prev.slice(-300), msg]);
      } catch {
        console.warn("[WS] Unparseable message:", event.data.slice(0, 80));
      }
    };
  }, [url, reconnectDelay]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const clearHistory = useCallback(() => {
    setMessageHistory([]);
    setLastMessage(null);
  }, []);

  return { isConnected, lastMessage, messageHistory, sendMessage, clearHistory };
}
